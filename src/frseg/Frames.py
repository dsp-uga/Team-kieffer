"""
			Author: Narinder Singh				Project: Cilia Segmentation			Date: 27 Feb 2019
			Course: CSCI 8360 @ UGA				Semester: Spring 2019				Module: Frames.py

Description: This module contains methods and classes for manipulating image frames.
"""

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as matplot

import Utilities

class FramesError(Exception): pass
class BadPathError(FramesError): pass
class UnexpectedDimensionsError(FramesError): pass

def computeMean(dir):
	"""
		Computes pixel means for the set of images in the given @directory. All the images in the directory must have the same resolution. The mean is computed per pixel and the results are returned as a numpy matrix.
		
		Note on space complexity: The method reads in one image at a time and processes it. In big-oh notation, the complexity is O(h.w).
	"""
	# Validate input directory
	if not os.path.isdir(dir): raise BadPathError("Path: " + dir + " is not a directory.")
	
	# Uninitialized matrix for aggregating pixel values. To be initialized when the first image is read in.
	temp = None
	
	# Number of images
	count = float(len(os.listdir(dir)))
	
	# Read each image and add to temp
	for fname in os.listdir(dir):
		if not Utilities.isImageFile(fname): continue
		fpath = os.path.join(dir, fname)
	
		# Read image as a numpy matrix
		img = Image.open(fpath)
		mat = np.asarray(img, np.int64)

		# Add to mean computation
		if temp is None:
			# Case 1: First image
			temp = mat.copy()
		else:
			# Case 2: Subsequent image.
			if mat.shape != temp.shape:
				# Make sure it's the same dimensions as earlier ones.
				raise UnexpectedDimensionsError("Image: " + fpath + " does not have the same dimensions as other images in the directory.")
			else:
				temp += mat

	if temp is None: return temp
	else: return temp / count


def computeVariance(dir):
	"""
		Computes variance by pixel for the set of images in the given @directory. All the images in the directory must have the same resolution. The variance is computed per pixel and the results are returned as a numpy matrix.
		
		Note on space complexity: The method reads in one image at a time and processes it. In big-oh notation, the complexity is O(h.w).
	"""
	# Validate input directory
	if not os.path.isdir(dir): raise BadPathError("Path: " + dir + " is not a directory.")
	# Empty Directory
	if not len(os.listdir(dir)): raise BadPathError("Path: "+ dir + " does not contain any image files.")

	# The mean matrix and a temp matrix for aggregating sum squared distances form the mean.
	mean = computeMean(dir)
	temp = np.zeros(mean.shape)
	
	# Number of images
	count = float(len(os.listdir(dir)))
	
	# Read each image and compound sum squared distances
	for fname in os.listdir(dir):
		fpath = os.path.join(dir, fname)

		# Read image as a numpy matrix
		img = Image.open(fpath)
		mat = np.asarray(img)

		# Add to sum squared distance computation
		temp += (mean - mat) * (mean - mat)
	
	return temp / count


if __name__ == '__main__':
	dir = "/Users/nsghumman/Documents/DataSciencePracticum/Team-kieffer/files/data/frames/fe397de12453e22d010132fa1927589b41566bdcf6796234e6606836f3c37927"
	mean = computeMean(dir)
	var = computeVariance(dir)
	
	matplot.scatter(xrange(len(var.flat)), var.flat)
	matplot.show()







