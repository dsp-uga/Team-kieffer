"""
			Author: Narinder Singh				Project: Cilia Segmentation			Date: 27 Feb 2019
			Course: CSCI 8360 @ UGA				Semester: Spring 2019				Module: Frames.py

Description: This module contains methods and classes for manipulating image frames.
"""

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as matplot

from Utilities import *
from Config import *

class FramesError(Exception): pass
class BadPathError(FramesError): pass
class UnexpectedDimensionsError(FramesError): pass

def computeMean(hash):
	"""
		Computes pixel means for the set of images in the given @directory. All the images in the directory must have the same resolution. The mean is computed per pixel and the results are returned as a numpy matrix.
		
		Note on space complexity: The method reads in one image at a time and processes it. In big-oh notation, the complexity is O(h.w).
	"""
	# Generate directory path
	dir = getVideoFramesDirectory(hash)
	
	# Validate input directory
	if not os.path.isdir(dir): raise BadPathError("Path: " + dir + " is not a directory.")
	
	# Uninitialized matrix for aggregating pixel values. To be initialized when the first image is read in.
	temp = None
	
	# Number of images
	count = float(len(os.listdir(dir)))
	
	# Read each image and add to temp
	for fname in os.listdir(dir):
		if not isImageFile(fname): continue
		fpath = os.path.join(dir, fname)
	
		# Read image as a numpy matrix
		img = Image.open(fpath)
		mat = np.asarray(img, np.int32)

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


def computeVariance(hash):
	"""
		Computes variance by pixel for the set of images in the given @directory. All the images in the directory must have the same resolution. The variance is computed per pixel and the results are returned as a numpy matrix.
		
		Note on space complexity: The method reads in one image at a time and processes it. In big-oh notation, the complexity is O(h.w).
	"""
	# Generate directory path
	dir = getVideoFramesDirectory(hash)
	
	# Validate input directory
	if not os.path.isdir(dir): raise BadPathError("Path: " + dir + " is not a directory.")
	# Empty Directory
	if not len(os.listdir(dir)): raise BadPathError("Path: "+ dir + " does not contain any image files.")

	# The mean matrix and a temp matrix for aggregating sum squared distances form the mean.
	mean = computeMean(hash)
	temp = np.zeros(mean.shape)
	
	# Number of images
	count = float(len(os.listdir(dir)))
	
	# Read each image and compound sum squared distances
	for fname in os.listdir(dir):
		if not isImageFile(fname): continue
		fpath = os.path.join(dir, fname)

		# Read image as a numpy matrix
		img = Image.open(fpath)
		mat = np.asarray(img)

		# Add to sum squared distance computation
		temp += (mean - mat) * (mean - mat)
	
	return temp / count


def plotVariance(hash, markCilia=True, save=False):
	"""
		This method plots variance of each pixel across the video frames. If markCilia flag is set, cilia and non-cilia pixels are identified.
		If the save flag is set, the resultant plot is saved to the data visuals directory in the Variance subdirectory. Refer Config module for the addresses.
	"""
	# Compute variance matrix and read cilia mask
	var = computeVariance(hash)
	mask = readMask(hash)

	# Create two matrices: one recording variances for cilia pixels and the other for cell and the background
	cilias = var * mask
	others = var * invertMask(mask)

	# Create variance scatterplot
	matplot.figure()
	matplot.scatter(xrange(cilias.size), cilias.flat, marker='+', color='red', label='cilia var')
	matplot.scatter(xrange(others.size), others.flat, marker='+', color='blue', label='others var')
	matplot.scatter(xrange(var.size), [var.mean()] * var.size, marker = '_', color='green', label='mean var')
	matplot.scatter(xrange(var.size), [var.mean() + var.std()] * var.size, marker='_', color='yellow', label='mean + 1 sigma' )
	matplot.legend(loc='best')
	
	# Save or display
	savedir = os.path.join(DATA_VISUALS_PATH, "Variance/")
	if save: matplot.savefig(os.path.join(savedir, hash + ".png"))
	else: matplot.show()
	matplot.close()


def plotCiliaCounts(hashes, percentages=False):
	"""
		Plot a scatterplot for the count of cilia cells in the given hashes. percentages flag dictates if the count should be transformed to percent cilia cells in the frame.
	"""
	# List to collect the results and a bar to track progress
	results = []
	bar = ProgressBar(max=len(hashes), message="Processing data ....")

	# Read each mask and count the number of cilia pixels
	for hash in hashes:
		mask = readMask(hash)
		elements, counts = np.unique(mask, return_counts=True)
		
		# Make a count-map and extract cilia count. The keys will simple be True and False where True stands for cilia pixels
		cmap = dict(zip(elements, counts))
		cilias = cmap.get(True, 0)
		
		# Collect result and update progress
		if percentages: results.append(cilias/float(mask.size))
		else: results.append(cilias)
		bar.update()
	
	label = "%age" if percentages else "counts"
	matplot.title("Cilia " + label)
	matplot.scatter(xrange(len(results)), results, marker='+', color='blue', label = label)
	matplot.scatter(xrange(len(results)), [mean(results)] * len(results), marker='_', color='red', label = "mean")
	matplot.legend(loc='best')
	matplot.show()


if __name__ == '__main__':
	# Quick testing etc.
	diri = "/Users/nsghumman/Documents/DataSciencePracticum/Team-kieffer/files/data/frames/"
	hash = "7167939da20844cd30f8e63009c73bd89dbb36e3ec38878f32f9781228c53e2b"
	temp = [ "cf621707b159de8b3e57f31ed68adbc5e239c41b389c0443c98d40d10e886e01",
			   "dbafcd86679aeada1a8d977d691b89089142cdae380a2f3158099db5db0b713e",
			   "ad6eac5d0cfc44219b69a507bc987d279568e54af6cb88b3682e26c8c4710970",
			   "94393237ac4081715d3443b15fadb274bd2095ac9e1992c8ac7e9560631a5f7b",
			   "77eb15c0e7e51dd9688a5e580b5681a2416fd8a7c82a5a7f6f4d5e2b68aeda0a",
			   "717f959cad7c9cbf084577ef236e0d39783e43994f1d56705bf1a33aa67d7965",
			   "7167939da20844cd30f8e63009c73bd89dbb36e3ec38878f32f9781228c53e2b",
			   "2237443684da4c50851e9c1ebc479859ec30bc52d5ee3a78266c9f07e8f17f91"
			 ]
			 
	hashes = readLines(TRAIN_FILE)
	for hash in hashes:
		plotVariance(hash, save = True)







