
from RayTracingUtilities import listify, delistify, makeHomogeneous, normalize, \
    listLength, listDot
from triangle import Triangle
from parallelogram import Parallelogram
from raytracefunc import rayTrace

import numpy as np
import matplotlib.pyplot as plot

class Scene :


    # The constructor that initializes a scene from its light position (a NumPy
    # column vector in homogeneous form), ambient light intensity, and
    # elements.

    def __init__( self, light, ambient, elements ) :
        self.lightPos = light
        self.ambient = ambient
        self.elements = elements



class Parallelogram :

    def __init__( self, ptA, ptB, ptC, red, green, blue, specular, shine, RefractiveIndex ) :

        # Store the color information.
        self.red = red
        self.green = green
        self.blue = blue
        self.specular = specular
        self.shine = shine
        self.RefractiveIndex = RefractiveIndex

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


    # Calculate the intersections between a set of rays and this parallelogram. See
    # above for a description of the arguments to and results from this method.
    def intersect( self, origins, directions ) :

        nRays = listLength( origins )

        # Transform rays to Cartesian canonical coordinates.
        canonicalOrigins = ( self.toCanonical @ origins )[0:3,:]
        canonicalDirections = ( self.toCanonical @ directions )[0:3,:]

        # Accumulate a mask for those rays that actually intersect the parallelogram.
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
        hitMask[canonicalHits[0, :] > 1.0] = False
        hitMask[canonicalHits[1, :] > 1.0] = False

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

class Ellipsoid :


    # Initialize an ellipsoid with the position of its center, its "radius" in
    # each dimension, and its color. Position is a 4-component NumPy column
    # vector giving the center point for the ellipsoid in homogeneous form; the
    # other arguments are real-valued scalars.

    def __init__( self, center, xRadius, yRadius, zRadius, red, green, blue, specular, shine, RefractiveIndex ) :

        # Save the color information.
        self.red = red
        self.green = green
        self.blue = blue
        self.specular = specular
        self.shine = shine
        self.RefractiveIndex = RefractiveIndex

        # The world-to-canonical coordinates transformation is the inverse of a
        # transformation that scales by the appropriate radii and then
        # translates to the new position -- i.e., a transformation that
        # first translates back to the canonical origin and then scales by the
        # reciprocals of the radii.
        self.toCanonical = np.array( [ [ 1.0/xRadius,  0.0,     0.0,       -center[0,0]/xRadius ],
                                       [   0.0,   1.0/yRadius,  0.0,       -center[1,0]/yRadius ],
                                       [   0.0,        0.0,  1.0/zRadius,  -center[2,0]/zRadius ],
                                       [   0.0,        0.0,     0.0,          1.0               ] ] )


    # Calculate intersections between this ellipsoid and a set of rays. See
    # above for a detailed description of the arguments to and results from
    # this method.

    def intersect( self, origins, directions ) :

        # Start by transforming all the rays to the ellipsoid's canonical
        # coordinate system. But I don't need the transformed values in
        # homogeneous form.
        canonicalOrigins = (self.toCanonical @ origins)[ 0:3, : ]
        canonicalDirections = (self.toCanonical @ directions)[ 0:3, : ]

        # A ray from origin <Rx,Ry,Rz> in direction (dx,dy,dz) intersects the
        # canonical sphere where
        # (dx^2 + dy^2 + dz^2) t^2 + 2(Rxdx + Rydy + Rzdz) t + (Rx^2 + Ry^2 + Rz^2 - 1) = 0.
        # Calculate the coefficients of this equation for each direction
        # vector.
        aValues = np.sum( canonicalDirections ** 2, 0 )
        bValues = 2.0 * np.sum( canonicalOrigins * canonicalDirections, 0 )
        cValues = np.sum( canonicalOrigins ** 2, 0 ) - 1.0

        # Use the quadratic formula to solve for t values where rays intersect
        # the canonical sphere. First I need to know where the rays intersect
        # the sphere at all, i.e., where the equation has real solutions.
        discriminants = bValues ** 2 - 4.0 * aValues * cValues
        hitMask = discriminants >= 0.0

        # Now, the result is infinity where there's no intersection or both
        # solutions to the quadratic are negative, the lower solution to the
        # quadratic if that solution is positive and real, and the higher
        # solution if the lower is negative but the higher is positive.
        nRays = listLength( directions )

        lowTs = np.full( nRays, np.inf )
        lowTs[hitMask] = ( -bValues[hitMask] - np.sqrt(discriminants[hitMask]) ) / ( 2.0 * aValues[hitMask] )
        lowMask = hitMask & ( lowTs > 0.0 )

        highTs = np.full( nRays, np.inf )
        highTs[hitMask] = ( -bValues[hitMask] + np.sqrt( discriminants[hitMask] ) ) / ( 2.0 * aValues[hitMask] )
        highMask = hitMask & ( lowTs <= 0.0 ) & ( highTs > 0.0 )

        finalTs = np.full( nRays, np.inf )
        finalTs[ lowMask ] = lowTs[ lowMask ]
        finalTs[ highMask ] = highTs[ highMask ]

        hitMask = lowMask | highMask                    # Visible intersections are where at least 1 t is positive

        # Calculate global intersection points for the intersections.
        finalPoints = np.ones( [ 4, nRays ] )
        finalPoints[ :, hitMask ] = origins[ :, hitMask ] + finalTs[ hitMask ] * directions[ :, hitMask ]

        # Calculate global normals at the intersections. Start with the local
        # intersection points in Cartesian form, which, thanks to the geometry
        # of the canonical sphere, are also local normals. Multiply those local
        # normals by the transpose of the global-to-canonical matrix, i.e., the
        # transpose of the inverse of the canonical-to-global matrix, in order
        # to put them into global coordinates, and finally normalize them.
        canonicalNormals = ( canonicalOrigins[ :, hitMask ] + finalTs[ hitMask ] * canonicalDirections[ :, hitMask ] )[ 0:3, : ]
        toGlobal = np.transpose( self.toCanonical[ 0:3, 0:3 ] )
        finalNormals = np.zeros( [ 4, nRays ] )
        finalNormals[ :, hitMask ] = makeHomogeneous( normalize( toGlobal @ canonicalNormals ), 0 )

        # All done. Return the t values, intersection points, and normals.
        return finalTs, finalPoints, finalNormals






def traceForColor( scene, origin, directions, maxDepth=1 ) :


    # Pull object colors out of the scene and into a NumPy matrix. Each object
    # has a column, with red, green, and blue coefficients of diffuse
    # reflection in the 1st, 2nd, and 3rd rows respectively. Similarly pull
    # coefficients of specular reflection and shininess exponents out of the
    # scene's objects.

    diffuseColors = np.array( [ [ e.red for e in scene.elements ],
                                [ e.green for e in scene.elements ],
                                [ e.blue for e in scene.elements ] ] )

    specularCoefficients = np.array( [ e.specular for e in scene.elements ] )
    shineExponents = np.array( [ e.shine for e in scene.elements ] )
    #gather all the refractive values of objects in scene
    refractiveIndices = np.array([e.RefractiveIndex for e in scene.elements])


    # Radiant intensities of the light for each color. These are more or less
    # arbitrary values, defined here to make them easy to change later.

    radiantIntensity = np.array( [[12.0], [12.0], [12.0]] )


    # Figure out how many pixels there will be in the image.

    nPixels = listLength( directions )


    # Trace the primary rays, i.e., the ones from the origin through each
    # pixel. Make masks for subsequent calculations based on where these rays
    # intersect or don't intersect anything.

    objects, hits, normals, hitTs = rayTrace( scene.elements, np.full( [ 4, nPixels ], origin ), directions )
    hitMask = objects >= 0
    missMask = objects < 0


    # Now that I know where rays really hit objects, replace all the -1 object
    # IDs where rays missed everything with dummy values of 0 so that all
    # object IDs will be valid object indices from here on.

    objects[ missMask ] = 0


    # Build object colors piece by piece, starting with everything being black.

    colors = np.zeros( [ 3, nPixels ] )


    # Add in the contribution of ambient light.

    ambients = np.full( [3, nPixels], scene.ambient )
    colors += ambients * diffuseColors[ :, objects ]


    # Figure out where shadows lie. Conceptually, do this by tracing rays from
    # each intersection of the original rays with a surface to the light,
    # taking each such ray that intersects any object to come from a point in
    # shadow. This assumes that the light is outside of the scene, i.e., that
    # it's not possible for a shadow ray to intersect an object after it passes
    # through the light. I also don't actually trace shadow rays from exactly
    # the original intersection points, but rather from those points displaced
    # slightly along the normals to the surfaces, in order to be sure a ray
    # doesn't intersect its own starting point due to roundoff.

    shadowOrigins = hits + 1.0e-6 * normals
    shadowDirections = scene.lightPos - shadowOrigins

    shadowObjects, shadowHits, shadowNormals, shadowTs = rayTrace( scene.elements, shadowOrigins, shadowDirections )



    litMask = hitMask & ( shadowTs > 1.0 )


    # Add the contribution of diffuse reflection to the colors.

    lightVectors = scene.lightPos - hits
    lightLengths = np.linalg.norm( lightVectors, axis=0 )

    irradiance = radiantIntensity / lightLengths ** 2

    unitLightVectors = normalize( lightVectors )
    lightCosines = np.maximum( listDot( normals, unitLightVectors ), 0.0)




    # Add the contribution of specular reflection, using the angle between the
    # halfway vector and the normal as the basis for scaling light intensity.

    viewVectors = normalize( -directions )
    halfways = normalize( lightVectors + viewVectors )
    halfwayCosines = np.maximum( listDot( halfways, normals ), 0.0 )


    # Pixels not occupied by any object receive a background color.

    bkgndColor = np.array( [ [0.019], [0.026], [0.019] ] )
    colors[ :, missMask ] = bkgndColor


    # All color calculations are finished, return the result.

    #Generating Refraction Rays
    refractionColors = np.zeros([3, nPixels])  # Initialize refraction color contribution
    # Check what surface is refractive using a mask. Values of 0 or less represent non-transparent objects.
    refractiveMask = (refractiveIndices[objects] < 4) #& (listDot(viewVectors, normals) > 0)
    rIndicies = 1/refractiveIndices[objects]
    #refractions = ((rIndicies*listDot(normals,viewVectors)-np.sqrt(1-((rIndicies)**2)*(1-(listDot(normals,viewVectors))**2)))*normals)-rIndicies*viewVectors




    if np.any(refractiveMask) and maxDepth > 0:
        refractiveOrigins = hits[:, refractiveMask] - 2.01*normals[:, refractiveMask] # establish origins of refraction

        #generate refraction rays using recursion. Refraction rays are called using expansions of snells law
        refractions = ((rIndicies * listDot(normals, viewVectors) -
            np.sqrt(1 - ((rIndicies) ** 2) * (1 - (listDot(normals, viewVectors)) ** 2))) * normals) - rIndicies * viewVectors
        refractiveDirections = refractions[:,refractiveMask] #establish direction vectors. Done by taking the refractions of the refractive mask.
        refractionColors[:,refractiveMask] = traceForColor(scene, refractiveOrigins, refractiveDirections, maxDepth - 1) #trace rays for color recursively

    #appending sets of colors together. Now including refraction colors
    colors[:, litMask] += (irradiance * lightCosines * diffuseColors[:, objects])[:, litMask]
    colors[:, litMask] += refractionColors[:,litMask]


    return colors




# A function that scales color intensities to lie between 0 and 1.
# Specifically, this function multiplies all color values by a scale factor
# designed to scale the top few percent (the exact percentage is set in the
# function) of distinct color values beyond 1, and it then clamps the scaled
# values to be no less than 0 and no more than 1. The argument to this function
# is a NumPy matrix representing a list of colors, the result is a similar list
# of adjusted colors.

def scaleColor( pixels ) :

    intensities = np.sort( np.ravel(pixels) )
    shiftedIntensities = np.concatenate( [ np.array( [-1.0] ), intensities[:-1] ] )
    distinctIntensities = intensities[ intensities != shiftedIntensities ]

    nDistinct = len( distinctIntensities )
    referenceIntensity = distinctIntensities[ int( nDistinct * 0.98 ) ]             # Top 2% of intensities will scale beyond 1

    scaledPixels = pixels / referenceIntensity
    return np.maximum( np.minimum( scaledPixels, 1.0 ), 0.0 )


def refractionTestscene():
    return Scene(np.array( [ [0.0], [10.0], [10.0], [1.0] ] ),0.0,
                 [Parallelogram(np.array( [ [2], [0], [-1], [1.0] ] ),
                              np.array( [ [2], [2], [-1], [1.0] ] ),
                              np.array( [ [-2], [0], [-1], [1.0] ] ),0.8,0.24,0.8,0.1,1, 1.54),
                  Ellipsoid(np.array([[0], [1.0], [-4.0], [1.0]]),
                            10.0, 1.0, 1.0,
                            1.0, 0.0, 0.0, 0.8, 1.0, 4.5)])

def flippedTestscene():
    return Scene(np.array( [ [0.0], [10.0], [10.0], [1.0] ] ),0.0,
                 [Parallelogram(np.array( [ [-2], [0], [-1], [1.0] ] ),
                              np.array( [ [2], [0], [-1], [1.0] ] ),
                              np.array( [ [-2], [2], [-1], [1.0] ] ),0.8,0.24,0.8,0.1,1, 1.54),
                  Ellipsoid(np.array([[0], [1.0], [-4.0], [1.0]]),
                            10.0, 1.0, 1.0,
                            1.0, 0.0, 0.0, 0.8, 1.0, 4.5)])
def refractiveSphereTestscene():
    return Scene(np.array( [ [0.0], [6.0], [4.0], [1.0] ] ),0.0,
                 [Ellipsoid(np.array([[0], [1.0], [-2.0], [1.0]]),
                            1.0, 1.0, 1.0,
                            0.1, 0.1, 0.1, 0.8, 1.0, 1.5),
                  Ellipsoid(np.array([[0.0], [1.0], [-6.0], [1.0]]),
                            1.0, 1.0, 0.8,
                            1.0, 0.0, 0.0, 0.8, 1.0, 4.5)])
scene = refractiveSphereTestscene()


# Define the focal point and image grid. For this ray tracer, the image grid
# is a square in the z = 0 plane, centered at the origin, and the focal point
# is somewhere on the positive z axis. This assumes scenes mostly lie in the
# z < 0 halfspace.

#Currently the viewer is encountering issues with moving away/getting close to the objects in a scene
focusZ = 0.65                           # Z coordinate of focal point
focus = np.array( [ [0.0], [0.0], [focusZ], [1.0] ] )

nPixels = 2000                          # Number of pixels in each dimension of the image
totalPixels = nPixels * nPixels         # Total number of pixels in the image

imageLeft = -1.0                        # X coordinate of left edge of image
imageRight = 1.0                        # Right edge of image
imageBottom = -1.0                      # Y coordinate of bottom of image
imageTop = 1.0                          # Top of image

halfPixel = ( imageRight - imageLeft ) / ( 2.0 * nPixels )      # Half the width or height of a pixel in world units


# Build a "list" of direction vectors for the rays. This is really a NumPy
# matrix with the direction vectors in its columns, as described in the
# "RayTracingUtilities" module.

imageXs = np.linspace( imageLeft + halfPixel, imageRight - halfPixel, nPixels )
imageYs = np.linspace( imageTop - halfPixel, imageBottom + halfPixel, nPixels )

gridXs, gridYs = np.meshgrid( imageXs, imageYs )
gridZs = np.full( totalPixels, focusZ )
gridWs = np.full( totalPixels, 0.0 )

directions = listify( [ gridXs - focus[0,0], gridYs - focus[1,0], -gridZs, gridWs ] )


# Trace the rays, and unpack the resulting colors into an image.

image = delistify( scaleColor( traceForColor( scene, focus, directions ) ), [ nPixels, nPixels ] )


plot.imshow( image )
plot.show()
