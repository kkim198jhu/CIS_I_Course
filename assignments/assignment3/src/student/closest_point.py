"""Closest-point queries against the triangle mesh."""

from typing import Tuple

import numpy as np

from data_types.nominal_types import vct3
from ..mesh import Mesh


def closest_point_on_triangle(p: vct3, a: vct3, b: vct3, c: vct3) -> vct3:
    """
    Return the closest point to p on the triangle with corners a, b, c.
    
    input: query point p, triangle corners a, b, c
    output: the point of the triangle (face, edge or vertex) closest to p
    """
    P = p.vec.ravel()
    A = a.vec.ravel()
    B = b.vec.ravel()
    C = c.vec.ravel()
    
    ab = B - A
    ac = C - A
    ap = P - A
    
    #Find the projection Q = A + u * ab + v * ac.
    edges = np.column_stack((ab, ac))
    uv, _, rank, _ = np.linalg.lstsq(edges, ap, rcond=None)
    u, v = uv
    
    #Checks if the point is on the triangle
    if rank == 2 and u >= 0 and v >= 0 and u + v <= 1:
        return vct3(A + u * ab + v * ac)
    
    
    best_point = A
    best_dist_sq = ap @ ap
    
    #Finds a candidate on each edge
    for start, end in ((A, B), (B, C), (C, A)):
        edge = end - start
        length_sq = edge @ edge
        
        if length_sq == 0:
            candidate = start
        else:
            t = ((P - start) @ edge) / length_sq
            t = np.clip(t, 0.0, 1.0)
            candidate = start + t * edge
        
        #Compare with other candiates and returns the best
        difference = P - candidate
        dist_sq = difference @ difference
        
        if dist_sq < best_dist_sq:
            best_point = candidate
            best_dist_sq = dist_sq
    
    return vct3(best_point)


    raise NotImplementedError("TODO: implement closest_point_on_triangle")


def closest_points_on_mesh(points: np.ndarray, mesh: Mesh) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    TODO: implement (vectorised; a per-triangle Python loop is far too slow).
    For each input point in CT coordinates, find the closest point on the mesh.
    
    input: points (K,3) in CT coordinates, mesh
    output: closest (K,3), triangle index (K,), distance (K,)
    """
    #Shapes it to be a triangle
    triangles = mesh.tri_vertices  
    
    #Gets the corners of every triangle
    A = triangles[:, 0, :] 
    B = triangles[:, 1, :]  
    C = triangles[:, 2, :]
    
    #Gets the edges
    ab = B - A
    ac = C - A
    
    #find the dot products
    ab_ab = np.sum(ab * ab, axis=1)
    ab_ac = np.sum(ab * ac, axis=1)
    ac_ac = np.sum(ac * ac, axis=1)
    
    #Check that it isn't just a line and that the two points are actually spread out
    det = ab_ab * ac_ac - ab_ac**2
    
    #Checks that they are actually valid or not (points too close to each other or not)
    valid = np.zeros(len(det), dtype=bool)
    for i in range(len(det)):
        if det[i] > 1e-12 * ab_ab[i] * ac_ac[i]:
            valid[i] = True
    
    bc = C - B
    # Gets a to b, b to c, and c to a
    edge_data = [(A, ab, ab_ab), (B, bc, np.sum(bc**2, axis=1)), (C, -ac, ac_ac)]
    
    #Intializes variables needed here
    closest = np.empty((len(points), 3))
    triangle_indices = np.empty(len(points), dtype=int)
    distances = np.empty(len(points))
    
    #For every input (points), project it onto every triangle plane
    for i, P in enumerate(points):
        #Projects the point p onto every triangle's plane
        ap = P - A
        ab_ap = np.sum(ab * ap, axis=1)
        ac_ap = np.sum(ac * ap, axis=1)
        
        #Finds out how much we move along ab and ac for each projection (valid can skip)
        u = np.divide(ac_ac * ab_ap - ab_ac * ac_ap, det, out=np.zeros_like(det, dtype=float), where=valid)
        v = np.divide(ab_ab * ac_ap - ab_ac * ab_ap, det, out=np.zeros_like(det, dtype=float), where=valid)
        
        #Lets u or v scale all 3 coordinates of its edge (think that each point has a scaled/weighted that can determine midpoitns)
        #This comes up with a 3d point
        #Q is the closest point on the triangle's plane surface
        Q = A + u[:, None] * ab + v[:, None] * ac
        
        #Checks which triangles contain Q doing 4 checks
        #1) Checks that is a valid triangle (look earlier)
        #2) u[j] >= 0: The weight for B is zero or positive.
        #3) v[j] >= 0: The weight for C is zero or positive
        #4) u[j] + v[j] <= 1: The weight left for A is zero or positive
        on_face = np.zeros(len(u), dtype=bool)
        for j in range(len(u)):
            if valid[j] and u[j] >= 0 and v[j] >= 0 and u[j] + v[j] <= 1:
                on_face[j] = True
        
        #Gets a copy of Q
        best_points = Q.copy()
        best_dist_sq = np.sum((P - Q)**2, axis=1)
        best_dist_sq[~on_face] = np.inf
        
        #Checks each edge at a time to find the closest one to point Q
        for start, edge, length_sq in edge_data:
            #Find the P's projection onto the edge’s line
            along_edge = np.sum((P - start) * edge, axis=1)
            t = np.divide(along_edge, length_sq, out=np.zeros_like(length_sq, dtype=float), where=length_sq > 0)
            
            #Turn t into a point and measure its distance
            t = np.clip(t, 0.0, 1.0)
            candidate = start + t[:, None] * edge
            candidate_dist_sq = np.sum((P - candidate)**2, axis=1)
            
            #If it is the best point and it's valid, then save it, else don't
            for j in range(len(on_face)):
                if not on_face[j] and candidate_dist_sq[j] < best_dist_sq[j]:
                    best_points[j] = candidate[j]
                    best_dist_sq[j] = candidate_dist_sq[j]
        
        #Save the winning point, triangle, and distance 
        best_index = np.argmin(best_dist_sq)
        closest[i] = best_points[best_index]
        triangle_indices[i] = best_index
        distances[i] = np.sqrt(best_dist_sq[best_index])
    
    return closest, triangle_indices, distances
