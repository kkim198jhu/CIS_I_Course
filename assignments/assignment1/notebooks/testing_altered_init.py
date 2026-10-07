import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src import simulation
from src import vct3, Rot, Frame

np.set_printoptions(precision=3, suppress=True)
runs = 500
all_run_errors = []

for i in range(runs):

    ptr_local_positions = [
        # TODO: at least 4 non-coplanar vct3 points, e.g. vct3(x, y, z)
        # optimizing tracker placement here, ideally want to maximize volume and disrupl symmery
        # TODO: math proff/paper as backing
        vct3( 80,  80,  80),
        vct3( 80, -60, -80),
        vct3(-80,  65, -80),
        vct3(-70, -80,  70)
    ]
    ptr_tip_approx = vct3(0, 0, 50)  # TODO: vct3, the tip's approximate position in the pointer's local frame
    
    cal_local_positions = [
        # TODO: at least 4 non-coplanar vct3 points
        # same idea here, avoiding symmetry
        vct3(-10, 35, -10),
        vct3(40, 40, 0),
        vct3(15, 10, 40),
        vct3(10, -20, 5)
    ]
    
    F_tracker_approx = Frame(Rot(), vct3(1000, 1000, 2200))  # TODO: Frame, tracker pose relative to the workspace
    F_cal_approx = Frame(Rot(), vct3(1200, 700, 400))      # TODO: Frame, calibration object pose relative to the workspace
    
    ptr_mb = simulation.define_marker_body("ptr_mb", ptr_local_positions)
    cal_mb = simulation.define_marker_body("cal_mb", cal_local_positions)
    
    cal_placed = simulation.place_marker_body(cal_mb, F_cal_approx)
    # The pointer's own body doesn't have a meaningful "workspace placement" yet
    # It moves every time it's hand-guided (Step 4-6). 
    # precision="precise" matches a robot-controlled pose rather than a coarsely, manually placed
    # static object (see place_marker_body's docstring).
    ptr_placed = simulation.place_marker_body(ptr_mb, Frame.eye(), precision="precise")
    
    ptr = simulation.define_pointer("ptr1", ptr_tip_approx, ptr_mb)
    trk = simulation.create_tracker("trk1", F_tracker_approx)
    
    def refine_marker_body(trk, placed, n_readings=100, passes=1):
        """
        Improve the estimated positions of a body's markers by averaging repeated tracker readings in the body's local coordinate frame
        
        input:
            trk (src.tracker.Tracker) - Tracker to read with
            placed (src.simulation.marker_body.PlacedBody) - PlacedBody whose body's nominal marker positions should be refined in place (via body.set_marker_positions(...))
            n_readings [int] - readings to average per pass
            passes[int] - how many times to repeat
        output: None (mutates placed.body's nominal_marker_positions in place)
        """
        
        #Go through 3 passes
        for pass_idx in range(passes):
            all_readings = []
            
            #Go through all the n_readings
            for _ in range(n_readings):
                #Gets the sample marker bodies and the observed markers and it's corresponding frame
                my_dict = simulation.sample_marker_bodies(trk, [placed])
                obs = my_dict[placed.body.name]
                frame = obs.frame
                
                #Have to get the inverse as we are going from the body to the tracker and not tracker to body
                to_body = frame.inv()
                
                local_positions = []
                reading = []
                
                #For each marker, find the local positions relative to the bodies' own positions.
                #If that is confusing think that to_body is the frame shift and p is the marker position. This finds each individual marker sphere!
                for marker in obs.markers:
                    local_positions.append(to_body * marker.p)
                
                #Then get it in x,y,z so we can average it and then update
                for p in local_positions:
                    reading.append ([p.x, p.y, p.z])
                
                all_readings.append(reading)
                
            #Get the averages of all the positions from all the passes and update the passes
            ave = np.mean(all_readings, axis=0)
            updated_positions = []
            for pos in ave:
                updated_positions.append(vct3(pos))
            placed.body.set_marker_positions(updated_positions)
            
            #tracking error over the passes
            actual_positions = placed.body.get_actual_marker_positions()
            mean_error = np.mean([
                (p_est - p_true.p).norm()
                for p_est, p_true in zip(updated_positions, actual_positions)
            ])
            
        
    #initila error for ptr:
    error = np.mean([
        (p_est - p_true.p).norm()
        for p_est, p_true in 
        zip(ptr_local_positions, ptr_placed.body.get_actual_marker_positions())])
    refine_marker_body(trk, ptr_placed)
    refine_marker_body(trk, cal_placed)
    
    
    N_OBSERVATIONS = 64 # TODO: double check the number of observations
    F_ptr_list = []
    F_cal_list = []
    sensed_tip_list = []
    for j in range(N_OBSERVATIONS):
        R_desired = Rot.xyz((j // 8) * np.pi/28, 0, (j % 8) * np.pi/4)  #Gets 64 spherical combinations with x-tilt 0 to 45 and full azimuth sweep
        hg = simulation.hand_guide_pointer(ptr, ptr_placed, cal_placed, R_desired)
        obs = simulation.sample_marker_bodies(trk, [ptr_placed, cal_placed])
        sensed = simulation.sense_tip(ptr, hg)
        
        F_ptr_list.append(obs["ptr_mb"].frame)
        F_cal_list.append(obs["cal_mb"].frame)
        sensed_tip_list.append(sensed)
        
    
    sensed_positions = []
    valid_F_ptr_list = []
    valid_F_cal_list = []

    for index, sensed in enumerate(sensed_tip_list):
        #No variable taken
        if sensed is None:
            continue

        position = sensed.vec.ravel()

        # Error/Outside range
        if np.array_equal(position, [-10000, -10000, -10000]):
            continue

        sensed_positions.append(position)
        valid_F_ptr_list.append(F_ptr_list[index])
        valid_F_cal_list.append(F_cal_list[index])

    #Find the covariance and at least 2 valid observations
    if len(sensed_positions) >= 2:
        sensed_tip_cov = np.cov(
            np.array(sensed_positions), rowvar=False, ddof=1
        )
    
    def compute_pivot_calibration(F_ptr_list, F_cal_list, sensed_positions):
        """
        Using Least squares, estimate the pointer's local tip position. Using fit residuals, measure covariance. 
        
        input:
            F_ptr_list - List[Frame], the pointer marker body's pose relative to
                the tracker for each observation j (Fptr,j)
            F_cal_list - List[Frame], the calibration object's pose relative to
                the tracker for each observation j (Fcal,j), same order/length
        output:
            p_tip - vct3, the calibrated tip position in the pointer's local frame
            cov_estimate - np.ndarray (3x3), an estimated covariance for p_tip
                (e.g. from the least-squares fit's residuals)
        """
        #Checks if the lists are the same length
        if (len(F_ptr_list) != len(F_cal_list) and len(F_cal_list) != len(sensed_positions)):
            raise ValueError("Lists are not the same length")
        
        #Checks if they are at least 3 observations as 2 observations could just be in the same plane
        if (len(F_ptr_list) < 3):
            raise ValueError("Lists aren't long enough and need more observations")
        
        tip_positions = []
        for index in range(len(sensed_positions)):
            #Get the sensor reading
            sensor_point = vct3(sensed_positions[index])
            
            #Move the point to tracker coords.
            tracker_point = F_cal_list[index] * sensor_point
            
            #Move to pointer coordinates
            pointer_point = F_ptr_list[index].inv() * tracker_point
            
            #Store the 3 coordinates
            tip_positions.append(pointer_point.vec.ravel())
            
        #Average estimated tip position
        p_tip = vct3(np.mean(tip_positions, axis=0))
        
        #Cov estimated tip position
        cov_estimate = (
            np.cov(np.array(tip_positions), rowvar=False, ddof=1)/len(tip_positions)
        )
        
        return p_tip, cov_estimate
        
        

    p_tip_est, p_tip_cov = compute_pivot_calibration(valid_F_ptr_list, valid_F_cal_list, sensed_positions)

    
    near_local_positions = [
        vct3(80, 160, 160),
        vct3(80, -160, -160),
        vct3(-80, 160, -160),
        vct3(-80, -160,  160)
    ]
    F_near_approx = Frame(Rot.xyz(0, 0, 0), vct3(650, 300, 300))
    near_mb = simulation.define_marker_body("near_mb", near_local_positions)
    near_placed = simulation.place_marker_body(near_mb, F_near_approx)
    refine_marker_body(trk, near_placed)
    
    thetas = [0, np.pi / 6, -np.pi / 6, np.pi / 3, -np.pi / 3, np.pi / 2, -np.pi / 2]
    offsets = [0, 100, -100, 200, -200]
    # TODO: the full grid above is large (7^3 * 5^3 = 42,875 combinations) --
    # choose a representative subset of (position, orientation) test poses
    # rather than every combination.
    #Building a list from the thetas and offsets above
    test_poses = []
    for theta in thetas:
        test_poses.append(Frame(Rot.xyz(theta, 0, 0), vct3(300, 300, 300)))
        for offset in offsets:
            test_poses.append(Frame(Rot.xyz(0, theta, 0), vct3(300+offset, 300, 300)))
            test_poses.append(Frame(Rot.xyz(0, 0, theta), vct3(300, 300+offset, 300)))
            test_poses.append(Frame(Rot.xyz(theta, theta, theta), vct3(300, 300, 300+offset)))
    #Add corners for better representation
    test_poses.append(Frame(Rot.xyz(0, 0, 0), vct3(100, 100, 100)))
    test_poses.append(Frame(Rot.xyz(0, 0, 0), vct3(500, 500, 500)))
    test_results = []  # list of (F_t_nominal, error_mm)
    for F_t_nominal in test_poses:
        # TODO, for each F_t_nominal:
        #   1. test_placed = simulation.place_marker_body(ptr_mb, F_t_nominal, precision="precise")
        test_placed = simulation.place_marker_body(ptr_mb, F_t_nominal, precision="precise")
        
        #   2. obs = simulation.sample_marker_bodies(trk, [test_placed, near_placed])  -- single reading
        obs = simulation.sample_marker_bodies(trk, [test_placed, near_placed])
        
        #   3. predict the tip position relative to near_placed using p_tip_est
        pre_tip = obs[near_mb.name].frame.inv() * (obs[ptr_mb.name].frame * p_tip_est)
        
        #   4. compute the TRUE tip position relative to near_placed using
        #      test_placed.actual_placement / near_placed.actual_placement (ground truth,
        #      for reporting only -- see markdown above)    
        true_tip_w = test_placed.actual_placement * ptr.actual_tip_position.p
        true_tip_n = near_placed.actual_placement.inv() * true_tip_w
        #   5. record the error and append (F_t_nominal, error) to test_results
        error = np.linalg.norm((pre_tip - true_tip_n).vec)
        test_results.append((F_t_nominal, error))
        
        
    # TODO: summarize test_results -- e.g. mean/max error, and how many poses
    # meet the assignment's ||delta p_t|| <= 2mm spec.
    # TODO: average summary statistics for multiple runs?
    '''
    errors = np.array([error for _, error in test_results])
    print("Number of test poses:", len(errors))
    print("Mean error (mm):", np.mean(errors))
    print("Maximum error (mm):", np.max(errors))
    print("Poses within 2 mm:", np.sum(errors <= 2), "/", len(errors))
    print("Poses over 2 mm:", np.sum(errors > 2))
    print("Standard deviation (mm):", np.std(errors, ddof=1))
    print("Median error (mm):", np.median(errors))
    print("95th percentile (mm):", np.percentile(errors, 95))
    print("RMSE (mm):", np.sqrt(np.mean(errors ** 2)))
    worst_index = np.argmax(errors)
    print("Worst pose error (mm):", test_results[worst_index][1])
    
    errors_45 = np.array([
        error for F, error in test_results
        if F.R.matrix[2, 2] >= np.cos(np.pi / 4)
    ])
    print("Number of test poses, with \npointer oriented within\n 45 degrees of vertical:", len(errors_45))
    print("Mean error (mm):", np.mean(errors_45))
    print("Maximum error (mm):", np.max(errors_45))
    print("Poses within 2 mm:", np.sum(errors_45 <= 2), "/", len(errors_45))
    '''
    all_run_errors.append(
        np.array([error for _, error in test_results])
    )

print("\n")
print("\n")
print("Average mean error (mm):", np.mean([np.mean(e) for e in all_run_errors]))
print("Average maximum error (mm):", np.mean([np.max(e) for e in all_run_errors]))
print("Average poses within 2 mm:", np.mean([np.sum(e <= 2) for e in all_run_errors]), "/ 114")
print("Average poses over 2 mm:", np.mean([np.sum(e > 2) for e in all_run_errors]), "/ 114")
print("Average standard deviation (mm):", np.mean([np.std(e, ddof=1) for e in all_run_errors]))
print("Average median error (mm):", np.mean([np.median(e) for e in all_run_errors]))
print("Average 95th percentile (mm):", np.mean([np.percentile(e, 95) for e in all_run_errors]))
print("Average RMSE (mm):", np.mean([np.sqrt(np.mean(e ** 2)) for e in all_run_errors]))
print("Worst error across all runs (mm):",  max(np.max(e) for e in all_run_errors))
print("Runs where all poses meet 2 mm:", sum(np.all(e <= 2) for e in all_run_errors), "/", runs)






