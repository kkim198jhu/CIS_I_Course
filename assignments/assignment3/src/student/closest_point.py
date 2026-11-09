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
    input: points (K,3) in CT coordinates, mesh
    output: closest (K,3), triangle index (K,), distance (K,)
    """
    raise NotImplementedError("TODO: implement closest_points_on_mesh")
