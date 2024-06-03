# A ray tracer for scenes consisting of multiple ellipsoids and triangles lit
# by a single white point light source and some amount of white ambient light.
#
# The overall program design is as follows:
#   - Scenes provide information about the position of the light source, the
#     intensity of ambient light, and the geometric objects in the scene.
#     The color and radiant intensity of the light source is set for all scenes
#     in the "traceForColor" function, while ambient light is always white,
#     with an irradiance set independently for each scene.
#   - Geometric objects (amethysts and ellipsoids) are instances of canonical
#     objects. They know their color, their transformation from global to
#     canonical coordinates, and how to detect intersections between themselves
#     and rays. The result of these intersection calculations is three pieces
#     of information: the global points at which intersections happen, the
#     global normals to the object at those points, and the t values at which
#     the intersections happen. Object colors are represented by 5 numbers,
#     namely the coefficients of diffuse reflection, a coefficient of specular
#     reflection for all colors, and a Phong shininess exponent.
#   - Pixel colors are represented in RGB, by NumPy column vectors.
#   - Points and vectors are also column vectors, in homogeneous form.
#   - Lists of points, vectors, or colors are NumPy matrices, which each point
#     etc. in a column. See comments in "RayTracingUtilities.py" for more on
#     this list representation.
#   - The main program sets up the viewer and scene, calls a "traceForColor"
#     function to build an image from the scene, and displays the image.
#   - The traceForColor function generates an image from a scene and rays into
#     it. Rays are represented by a single origin point and a list of direction
#     vectors; the resulting scene is a list of pixel colors. This function
#     calls a more general "rayTrace" function that returns information about
#     the first intersection on each of multiple rays; "traceForColor" uses
#     this information to trace shadow rays (again by calling "rayTrace") and
#     do Phong shading for each pixel in the image.
#   - Because the shading model can now produce color intensities outside the
#     0 to 1 range allowed in images, a "scaleColor" function adjusts the color
#     values in the final image to the required range.

# History:
#
#   October 2021 -- Created by Doug Baldwin as a demonstration for Math 384.


from RayTracingUtilities import listify, delistify, makeHomogeneous, normalize, \
    listLength, listDot
from triangle import Triangle
from parallelogram import Parallelogram
from raytracefunc import rayTrace

import numpy as np
import matplotlib.pyplot as plot




# A class that represents scenes. This class serves only to store and provide
# access to the information about the scene, with a constructor as an easy way
# to initialize scenes. Attributes that hold the scene information are...
#   - The position of the light source, in attribute "lightPos"
#   - The intensity of ambient light, in attribute "ambient"
#   - The list of geometric objects in the scene, in attribute "elements"

class Scene :


    # The constructor that initializes a scene from its light position (a NumPy
    # column vector in homogeneous form), ambient light intensity, and
    # elements.

    def __init__( self, light, ambient, elements ) :
        self.lightPos = light
        self.ambient = ambient
        self.elements = elements




# Classes that represent geometric objects. All geometric objects have the
# following attributes and methods:
#   - Color information, consisting of...
#       o The coefficients of diffuse reflection for red, green, and blue
#         light, in attributes "red," "green," and "blue" respectively
#       o The coefficient of specular reflection for all colors of light, in
#         attribute "specular"
#       o A shininess exponent for Phong's lighting model, in attribute "shine"
#   - An "intersect" method that takes lists of ray origin points and ray
#     directions, in the global coordinate system, as its arguments, and
#     determines where those rays intersect the object. This method returns
#     three values, namely...
#       o A NumPy row vector of ray "t" parameters at which intersections
#         happen. These values are infinity for any rays that don't intersect
#         the object.
#       o A list of global points where the intersections happen. These are
#         undefined wherever the corresponding t value is infinite.
#       o A list of global normal vectors to the intersections. Undefined where
#         t values are infinite.


# A class that generates Amethyst crystals. Taking an input
class Amethyst:
    #define 14 pts to construct shape of amehtyst
    def __init__( self,
                  pt1, pt2, pt3, radius, height,
                  red, green, blue, specular, shine ):
        # Store the color information.
        self.red = red
        self.green = green
        self.blue = blue
        self.specular = specular
        self.shine = shine
        # Generate series of planes that make up amethyst. This code follows a pattern such that
        # points exist as such:
        #           pt13
        #       pt4     pt2
        #       pt3     pt1
        #           pt14 to generate

        self.PlaneA = Parallelogram(pt1, pt2, pt3)
        self.UpperA = Triangle(pt2, pt13, pt4)
        self.LowerA = Triangle(pt1, pt3, pt14)
        self.PlaneB = Parallelogram(pt3, pt4, pt5)
        self.UpperB = Triangle(pt4, pt13, pt6)
        self.LowerB = Triangle(pt3, pt5, pt14)
        self.PlaneC = Parallelogram(pt5, pt6, pt7)
        self.UpperC = Triangle(pt6, pt13, pt8)
        self.LowerC = Triangle(pt5, pt7, pt14)
        self.PlaneD = Parallelogram(pt7, pt8, pt9)
        self.UpperD = Triangle(pt8, pt13, pt10)
        self.LowerD = Triangle(pt7, pt9, pt14)
        self.PlaneE = Parallelogram(pt9, pt10, pt11)
        self.UpperE = Triangle(pt10, pt13, pt12)
        self.LowerE = Triangle(pt9, pt11, pt14)
        self.PlaneF = Parallelogram(pt11, pt12, pt1)
        self.UpperF = Triangle(pt12, pt13, pt2)
        self.LowerF = Triangle(pt11, pt1, pt14)

        self.toCanonical = np.array( [ [ 1.0, 0.0, 0.0, 0.0],
                                    [   0.0, 1.0, 0.0, 0.0],
                                    [   0.0, 0.0, 1.0, 0.0],
                                    [   0.0, 0.0, 0.0, 1.0] ] )
        self.normal = np.array([[0.0],[1.0],[0.0],[0.0]])

    # intersect methods calls the raytrace function. An apt description is found in
    # the raytacefunc.py file.
    def intersect(self, origins, directions):
        amethystObjects = [self.PlaneA, self.UpperA, self.LowerA, self.PlaneB, self.UpperB, self.LowerB,
                           self.PlaneC, self.UpperC, self.LowerC, self.PlaneD, self.UpperD, self.LowerD,
                           self.PlaneE, self.UpperE, self.LowerE, self.PlaneF, self.UpperF, self.LowerF]
        #amethystObjects = [self.PlaneA, self.UpperA, self.LowerA]
        amethystObjects, finalPoints, finalNormals, finalTs = rayTrace(amethystObjects, origins, directions)
        return finalTs, finalPoints, finalNormals



# A class that represents ellipsoids.
class Ellipsoid :


    # Initialize an ellipsoid with the position of its center, its "radius" in
    # each dimension, and its color. Position is a 4-component NumPy column
    # vector giving the center point for the ellipsoid in homogeneous form; the
    # other arguments are real-valued scalars.

    def __init__( self, center, xRadius, yRadius, zRadius, red, green, blue, specular, shine ) :

        # Save the color information.
        self.red = red
        self.green = green
        self.blue = blue
        self.specular = specular
        self.shine = shine

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







# The function that traces a set of rays from a single origin through a scene,
# returning a list of pixel colors for an image of that scene.

def traceForColor( scene, origin, directions ) :


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


    # Radiant intensities of the light for each color. These are more or less
    # arbitrary values, defined here to make them easy to change later.

    radiantIntensity = np.array( [[10.0], [10.0], [10.0]] )


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

    colors[ :, litMask ] += ( irradiance * lightCosines * diffuseColors[ :, objects ] )[ :, litMask ]


    # Add the contribution of specular reflection, using the angle between the
    # halfway vector and the normal as the basis for scaling light intensity.

    viewVectors = normalize( -directions )
    halfways = normalize( lightVectors + viewVectors )
    halfwayCosines = np.maximum( listDot( halfways, normals ), 0.0 )
    colors[ :, litMask ] += ( irradiance * specularCoefficients[objects] * halfwayCosines ** shineExponents[objects] )[ :, litMask ]


    # Pixels not occupied by any object receive a background color.

    bkgndColor = np.array( [ [0.007], [0.014], [0.08] ] )
    colors[ :, missMask ] = bkgndColor


    # All color calculations are finished, return the result.

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






# A scene consisting of a single matte white amethyst, standing upright
# in front of the viewer.  This is the most basic test. Functions called here are the
#amethyst and scene functions.

def oneAmethystScene() :
    scaleval=0.5
    return Scene( np.array( [ [0.0], [5.0], [10.0], [1.0] ] ), 0.0,
                   [ Amethyst( np.array( [ [0.25/np.sqrt(3)], [-1.67], [-0.75], [1.0] ] ),
                              np.array( [ [0.25/np.sqrt(3)], [1.67], [-0.75], [1.0] ] ),
                              np.array( [ [-0.25/np.sqrt(3)], [-1.67], [-0.75], [1.0] ] ),
                              np.array( [ [-0.25/np.sqrt(3)], [1.67], [-0.75], [1.0] ] ),
                              np.array( [ [-np.sqrt((0.25/np.sqrt(3))**2+(0.75)**2)], [-1.67], [-1.0], [1.0] ] ),
                              np.array( [ [-np.sqrt((0.25/np.sqrt(3))**2+(0.75)**2)], [1.67], [-1.0], [1.0] ] ),
                              np.array( [ [-0.25/np.sqrt(3)], [-1.67], [-1.25], [1.0] ] ),
                              np.array( [ [-0.25/np.sqrt(3)], [1.67], [-1.25], [1.0] ] ),
                              np.array( [ [0.25/np.sqrt(3)], [-1.67], [-1.25], [1.0] ] ),
                              np.array( [ [0.25/np.sqrt(3)], [1.67], [-1.25], [1.0] ] ),
                              np.array( [ [np.sqrt((0.25/np.sqrt(3))**2+(0.75)**2)], [-1.67], [-1.0], [1.0] ] ),
                              np.array( [ [np.sqrt((0.25/np.sqrt(3))**2+(0.75)**2)], [1.67], [-1.0], [1.0] ] ),
                              np.array([[0.0], [2.5], [-1.0], [1.0]]),
                              np.array( [ [0.0], [-2.5], [-1.0], [1.0] ] ),
                              1,1,1,.45,1) ] )


# The main program. This creates a scene to render, defines the viewer's
# position and rays from that point through the pixels of a virtual image grid,
# generates an image from that information, and finally displays the image.

# Set up the scene.

scene = oneAmethystScene()


# Define the focal point and image grid. For this ray tracer, the image grid
# is a square in the z = 0 plane, centered at the origin, and the focal point
# is somewhere on the positive z axis. This assumes scenes mostly lie in the
# z < 0 halfspace.

#Currently the viewer is encountering issues with moving away/getting close to the objects in a scene
focusZ = 0.45                           # Z coordinate of focal point
focus = np.array( [ [0.0], [0.0], [focusZ], [1.0] ] )

nPixels = 512                           # Number of pixels in each dimension of the image
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
