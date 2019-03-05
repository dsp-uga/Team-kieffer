"""
			Author: Narinder Singh				Project: Cilia Segmentation			Date: 27 Feb 2019
			Course: CSCI 8360 @ UGA				Semester: Spring 2019				Module: Frames.py

Description: This module contains statistical methods for manipulating image frames.
"""

import os
import math
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


def plotHeatMapVsMask(hash, sigma=0, save=False):
	"""
		Make visuals with thresholding for the video against the given hash. The format for @threshold is 'mean+Xsigma' where X is the number of standard deviations - a single digit non-negative integer. Each visual has 2 subplots:
		
			1. The heat-map of pixel variance
			2. The mask
	"""
	# Compute Variance matrix and extract its dimensions
	var = computeVariance(hash)
	rows, cols = var.shape
	
	# Apply threshold
	result = var * (var > var.mean() + sigma*var.std())

	# Make room for subplots
	fig, axes = matplot.subplots(nrows=1, ncols=2)
	(heatmap, mask) = axes
	
	# Variance heatmap
	heatmap.imshow(result, cmap='hot')
	
	# The mask
	mask.imshow(readMask(hash))

	# Save figure
	if save:
			savedir = os.path.join(DATA_VISUALS_PATH, "Thresholding/")
			fig.savefig(os.path.join(savedir, hash + ".png"))
			matplot.close()
	else:
		matplot.show()


def computeIoU(predicted, expected):
	"""
		Computes the Intersection over Union metric for two binary masks.
	"""
	intersection = predicted * expected
	union = predicted + expected
	
	# Re-adjust union back to [0, 1] scale and return the result.
	union[union == 2] = 1
	return float(sum(intersection.flat)) / (sum(union.flat) or 1)


def applyMeanThreshold(mat, sigma=0):
	"""
		Apply mean threhsold to the matrix. Optionally, one can move the threshold up or down @sigma standard deviations. The threshold excludes mean.
	"""
	result = mat * (mat > mat.mean() + sigma*mat.std())
	return result


def eval(hashes, sigma=0):
	"""
		Predict cilia masks using mean thresholding and evaluate them with the IoU metric returning the mean IoU score across the dataset. @sigma is an optional parameter signifying the number of standard deviations to go above or below the mean for threhsolding.
	"""
	# Initialize a list to store scores and a bar to keep track of progress.
	scores = []
	bar = ProgressBar(max=len(hashes), message="Computing IoUs ...")
	
	# Read each set of frames, predict mask and update the scores list.
	for hash in hashes:
		# Compute variance and threshold matrix to generate the mask.
		var = computeVariance(hash)
		result = applyMeanThreshold(var, sigma)
		
		# Binarize the mask and compute and append the IoU score.
		mask = result != 0
		iou = computeIoU(mask, readMask(hash))
		scores.append(iou)
		bar.update()

	# Return just the mean IoU of the dataset.
	return mean(scores)


def makePredictions(hashes, sigma=0):
	"""
		Predict (and save) cilia mask using variance thresholding. Default threhsold is the mean; @sigma is the number of standard deviations to go above the mean. The resultant greyscale is a single channel grayscale with 2 for cilia pixels and 0 otherwise. Observe the save directory at the end of this method and refer Config file to find out what it is.
	"""
	# Initialize a bar to display progress
	bar = ProgressBar(max=len(hashes), message = "Computing masks ...")
	
	# Read each 'video'
	for hash in hashes:
		# Compute variance and make prediction by thresholding
		var = computeVariance(hash)
		result = applyMeanThreshold(var, sigma)
		
		# Alter the mask to have 2s for cilia pixels and 0s otherwise
		mask = result != 0
		mask = mask.astype(np.uint8) * 2

		# Save prediction
		im = Image.fromarray(mask)
		im.save(os.path.join(PREDICTIONS_DEST_PATH, hash + ".png"))
		bar.update()


if __name__ == '__main__':
	# Quick testing etc.
	hashes = readLines(TEST_FILE)
	makePredictions(hashes)










