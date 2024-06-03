#   - The "rayTrace" function takes a list of geometric objects and a set of rays
#     as arguments, and returns 4 lists describing the first intersection each
#     ray has with an object in the scene: the index of the object intersected,
#     or -1 if there is no intersection, the global point at which the
#     intersection happens, the global normal to the object at that point, and
#     the t value at which the intersection happens. The rays given to
#     "rayTrace" are equal-length lists of origin points and direction vectors,
#     i.e., "rayTrace" does not assume all rays start at the same place.

import numpy as np
from RayTracingUtilities import listLength


# The function that traces arbitrary rays through a set of objects, returning
# geometric information (object intersected, intersection point, normal vector,
# and t parameter) about intersections rather than colors.

def rayTrace( objects, origins, directions ) :


    # This function accumulates object indices, intersection points, normals,
    # and corresponding t values in "lists" that it updates with intersection
    # information about each object in turn. Initialize those lists to "no
    # object" values: indices to -1, t values to infinity, and points and
    # normals to values that are homogeneous points and vectors respectively,
    # but otherwise arbitrary.

    nRays = listLength( origins )

    ts = np.full( nRays, np.inf )
    indices = np.full( nRays, -1 )
    hitPoints = np.ones( [ 4, nRays ] )
    hitNormals = np.zeros( [ 4, nRays ] )


    # Loop through the geometric objects, intersecting the rays with each and
    # updating t values, indices, intersection points, and normals wherever an
    # intersection happens at a smaller t value than previously seen.

    index = 0                               # Index of current object
    for obj in objects :

        newTs, newPoints, newNormals = obj.intersect( origins, directions )

        mask = newTs < ts
        ts[ mask ] = newTs[ mask ]
        indices[ mask ] = index
        hitPoints[ :, mask ] = newPoints[ :, mask ]
        hitNormals[ :, mask ] = newNormals[ :, mask ]

        index += 1


    # The results have now accumulated in the lists of indices, points, and
    # normals

    return indices, hitPoints, hitNormals, ts