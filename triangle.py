# Adapted by Frank Bubbico (February 2024) from Doug Baldwin
# The purpose of this file is to provide triangle objects for varying purposes.

# This triangle class is designed to contain no color information, only spatial information. This allows for the
# object to be called later as a piece of the amethyst class. This will also be used in other object classes in a
# similar way. This class is not unlike the first parallelogram class from parallelogram.py. The code varies in
# generating the shape from the E1/E2 vectors.

import numpy as np
from RayTracingUtilities import  normalize, \
    listLength

class Triangle :


    # In addition to the attributes that all geometric objects have, triangles
    # also store a unit-length normal column vector, in homogeneous global
    # coordinates, in attribute "normal".


    # Initialize a triangle with its vertices (NumPy column vectors, in
    # homogeneous form) and color information. Vertices should be given in
    # some counterclockwise order as seen from the side of the triangle that
    # the normal points to.

    def __init__( self, ptA, ptB, ptC) :



        # Initialize the transformation matrix. Build the canonical-to-global
        # transformation matrix described in Baldwin & Weber, then have NumPy
        # invert it.
        E1 = np.ravel( ptB - ptA )
        E2 = np.ravel( ptC - ptA )
        N = np.cross( E1[0:3], E2[0:3] )
        free = np.array( [ 1.0, 0.0, 0.0, 0.0 ] ) if abs( N[0] ) > abs( N[1] ) and abs( N[0] ) > abs( N[2] ) \
               else np.array( [ 0.0, 1.0, 0.0, 0.0 ] ) if abs( N[1] ) > abs( N[0] ) and abs( N[1] ) > abs( N[2] ) \
               else np.array( [ 0.0, 0.0, 1.0, 0.0 ] )

        self.toCanonical = np.linalg.inv( np.transpose( np.array( [ E1, E2, free, np.ravel(ptA) ] ) ) )

        # Store the normal vector, based on the one computed as part of finding
        # the transformation matrix. Note that what I found earlier is a row
        # vector in Cartesian form, not a column vector in homogeneous form,
        # nor necessarily unit length.
        self.normal = normalize( np.array( [ [N[0]], [N[1]], [N[2]], [0.0] ] ) )


    # Calculate the intersections between a set of rays and this triangle. See
    # above for a description of the arguments to and results from this method.

    def intersect( self, origins, directions ) :

        nRays = listLength( origins )

        # Transform rays to Cartesian canonical coordinates.
        canonicalOrigins = ( self.toCanonical @ origins )[0:3,:]
        canonicalDirections = ( self.toCanonical @ directions )[0:3,:]

        # Accumulate a mask for those rays that actually intersect the triangle.
        # Simultaneously accumulate a vector of t values at which those
        # intersections happen. Initially I consider all rays to be potential
        # intersectors, but the t values to be infinite.
        hitMask = np.full( nRays, True )
        ts = np.full( nRays, np.inf )

        # Any rays parallel to the xy plane cannot be intersectors.
        hitMask[ canonicalDirections[2,:] == 0.0 ] = False

        # Any rays that intersect the xy plane behind their starting points
        # (i.e., at negative t values) aren't really intersectors either.
        ts[ hitMask ] = -canonicalOrigins[ 2, hitMask ] / canonicalDirections[ 2, hitMask ]
        hitMask[ ts < 0.0 ] = False

        # Finally, intersections with the xy plane but outside the canonical
        # triangle don't intersect that triangle either. An intersection is
        # outside the canonical triangle if its x coordinate is negative, its
        # y coordinate is negative, or the sum of its x and y coordinates is
        # greater than 1.
        canonicalHits = np.ones( [ 3, nRays ] )
        canonicalHits[:,hitMask] = canonicalOrigins[:,hitMask] + ts[hitMask] * canonicalDirections[:,hitMask]
        hitMask[ canonicalHits[0,:] < 0.0 ] = False
        hitMask[ canonicalHits[1,:] < 0.0 ] = False
        hitMask[ canonicalHits[0,:] + canonicalHits[1,:] > 1.0 ] = False

        # The t values I actually want to return are the ones where a ray
        # intersects the canonical triangle; all other ts should be infinite.
        finalTs = np.full( nRays, np.inf )
        finalTs[ hitMask ] = ts[ hitMask ]

        # Build a list of world-coordinate points at which the rays intersect
        # this triangle.
        finalPoints = np.ones( [ 4, nRays ] )
        finalPoints[ :, hitMask ] = origins[ :, hitMask ] + finalTs[hitMask] * directions[ :, hitMask ]

        # The world-coordinate normals to the intersection points are all just
        # this triangle's normal:
        finalNormals = np.full( [ 4, nRays ], self.normal )

        # All done. Return t values, intersection points, and normals.
        return finalTs, finalPoints, finalNormals
