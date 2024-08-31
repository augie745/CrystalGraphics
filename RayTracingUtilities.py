
# History:
#
#   July 2021 -- Created by Doug Baldwin.
#   April 2024 -- Updated by Frank Bubbico.

# The purpose of this file is to contain functions that are primarily numpy based and
# are repeatedly used throughout other files in this branch.

import numpy as np

# I use NumPy matrices to represent "lists" of points, vectors, colors, and
# pretty much any other multi-component value. The idea behind all such lists
# is that each column of the NumPy matrix represents one of the points,
# vectors, etc. This happens to be an ideal representation if I need to
# transform all elements of the list with a transformation that is represented
# by a matrix, or if the list was produced by applying such a transformation.

# Create a list as described above from a list of components of the points,
# vectors, etc. The argument to this function is a list of NumPy values,
# all of which "ravel" to vectors of the same length. Each of those vectors
# provides one component of the things from which I'm making a list.

def listify( components ) :


    # I make the list by flattening all the components into row vectors, then
    # stacking them as rows of the result.

    flatComponents = [ np.ravel(c) for c in components ]
    return np.stack( flatComponents, 0 )




# Unpack a list as described above into a higher-dimensional object. Each item
# in the list (i.e., each column in the NumPy matrix) becomes a sequence along
# the last axis (i.e., a "row") of the result. Clients specify the number of
# earlier axes and their sizes.

def delistify( values, sizes ) :


    # Transpose the "list" to get the components of each item beside each other
    # in the order NumPy iterates over matrices, then reshape to the requested
    # shape.

    resultShape = sizes + [ -1 ]
    return np.reshape( np.transpose( values ), resultShape )




# Find the length of a list as described above.

def listLength( theList ) :


    # The length of the list is just the number of columns in the NumPy matrix.

    return np.shape( theList )[ 1 ]




# Given two lists, as described above, of vectors, compute their dot products.
# The result is a NumPy row vector of the products. The input lists must be the
# same size.

def listDot( vectors1, vectors2 ) :
    return np.sum( vectors1 * vectors2, 0 )




# Given two lists, as described above, of vectors in homogeneous form, compute
# a list of their cross products.

def listCross( vectors1, vectors2 ) :
    return np.array( [ vectors1[1,:] * vectors2[2,:] - vectors1[2,:] * vectors2[1,:],
                       vectors1[2,:] * vectors2[0,:] - vectors1[0,:] * vectors2[2,:],
                       vectors1[0,:] * vectors2[1,:] - vectors1[1,:] * vectors2[0,:],
                       np.zeros( listLength(vectors1) ) ] )




# Given a NumPy matrix that represents a list as described above of Cartesian
# points or vectors, return a new list of the same points or vectors in
# homogeneous form. Put the points or vectors into homogeneous form by
# appending a client-specified value onto the end of each (that value should be
# either 0 or 1).

def makeHomogeneous( pointsOrVectors, typeValue ) :


    # Generate as many copies of the final component as there are points or
    # vectors, then concatenate them on to the ends of those points/vectors.

    nItems = listLength( pointsOrVectors )
    fourthComponents = np.full( [1,nItems], typeValue )

    return np.concatenate( [pointsOrVectors, fourthComponents] )




# Given a list, as described above for "listify" and "delistify," of vectors,
# return a new list containing unit vectors parallel to the originals.

def normalize( vectors ) :
    return vectors / np.linalg.norm( vectors, axis=0 )




# Create an image array in which all pixels are a given color. Clients
# specify the color as an RGB triple in a NumPy 1-dimensional array. Clients
# also provide the number of rows and columns of pixels in the resulting image.
# Some of my ray tracers select between images of this form to get pixel colors
# for a final image.

def makeColorImage( color, nRows, nCols ) :
    return np.tile( color, [nRows, nCols, 1] )


# Further Developments

# Given a numpy array of vectors with an extra dimension of size 1, return the information
# contained within the array in the form of a [4,:] matrix for use of vector calculation.
# The purpose this function serves is to correct issues when designing matrices out of data that
# is already a numpy array like the instance in line 323 of RT3_ExtraordinaryRefraction.py.
def dimensionate(vectors):
    return np.array([vectors[:,0,0],vectors[:,1,0],vectors[:,2,0],vectors[:,3,0]])

# Given a list of 4x4 matrices (indexed [x,0:3,0:3]) and a list of vectors of length x,
# return a "matrix @ vector" calculation for each vector in the list.
def listMatricesMultiplication(matrices, vectors):
    return np.array([matrices[:,0,0]*vectors[0,:]+matrices[:,0,1]*vectors[1,:]+matrices[:,0,2]*vectors[2,:]+matrices[:,0,3]*vectors[3,:],
                     matrices[:,1,0]*vectors[0,:]+matrices[:,1,1]*vectors[1,:]+matrices[:,1,2]*vectors[2,:]+matrices[:,1,3]*vectors[3,:],
                     matrices[:,2,0]*vectors[0,:]+matrices[:,2,1]*vectors[1,:]+matrices[:,2,2]*vectors[2,:]+matrices[:,2,3]*vectors[3,:],
                     matrices[:,3,0]*vectors[0,:]+matrices[:,3,1]*vectors[1,:]+matrices[:,3,2]*vectors[2,:]+matrices[:,3,3]*vectors[3,:]])

# Simple function designed to return an inversed list of matrices. This typically yields a list of the form
# required in the function shown above.
def listMatrixInverse(matrix):
    return np.array([np.linalg.inv(matrix[:])])
