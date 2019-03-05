"""
			Author: Narinder Singh				Project: Cilia Segmentation			Date: 27 Feb 2019
			Course: CSCI 8360 @ UGA				Semester: Spring 2019				Module: Utilities.py

Description: This module contains methods and classes that make life easier.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as matplot
from scipy.misc import imsave
from PIL import Image
from Config import *

MASKS_PATH = os.path.join(DATA_FILES_PATH, "masks/")
LIT_MASKS_PATH = os.path.join(MASKS_PATH, "lit/")
FRAMES_PATH = os.path.join(DATA_FILES_PATH, "data/frames")

# Stretching constant for masks to scale the range of grayscales from [0, 2] to [0, 255]
MASK_STRETCHING_CONSTANT = 127.5

class UtilitiesError(Exception): pass
class BadHashError(UtilitiesError): pass


class ProgressBar:
	"""
		A handrolled implementation of a progress bar. The bar displays the progress as a ratio like this: (1/360).
	"""

	def __init__(self, max = 100, message = "Initiating ....."):
		"""
			Initialize the bar with the total number of units (scale).
		"""
		self.max = max
		self.current = 0
		print message + '\n'

	def update(self, add = 1):
		"""
			Record progress.
		"""
		self.current += add
		self._clear()
		self._display()

	def _display(self):
		"""
			Print the completion ratio on the screen.
		"""
		print "(" + str(self.current) + "/" + str(self.max) + ")"

	def _clear(self):
		"""
			Erase the old ratio from the console.
		"""
		sys.stdout.write("\033[F")
		sys.stdout.flush()


def flen(filename):
	"""
		File LENgth computes and returns the number of lines in a file. @filename <string> is path to a file. This is an epensive method to call for the whole file is read to determine the number of lines.
		
		returns: <integer> line count
	"""
	# Read and count lines.
	with open(filename, 'r') as infile:
		return sum((1 for line in infile))


def isImageFile(fpath):
	"""
		Returns whether or not the given path or filename is for an image file. The method is crude at the moment and just checks for some popular formats.
	"""
	path, fname = os.path.split(fpath)
	if fname.endswith(("png", "jpeg", "gif", "tiff", "bmp")): return True
	else: return False


def invertMask(mask):
	"""
		Inverts a numpy binary mask.
	"""
	return mask == False


def readMask(hash, binarize=True):
	"""
		Reads the mask for the given hash and if binarize flag is set, makes the mask binary (True/False : Cilia/Not-cilia)
	"""
	fpath = os.path.join(MASKS_PATH, hash + ".png")
	if not os.path.isfile(fpath): raise BadHashError("Hash: " + hash + " does not exist OR does not have a mask against it.")

	img = Image.open(fpath)
	mat = np.asarray(img, np.int32)
	mat.setflags(write=1)
	
	if binarize:
	 	ciliaMask = mat == CILIA_GRAYSCALE
	 	backgroundMask = invertMask(ciliaMask)
	 	mat[ciliaMask] = True
	 	mat[backgroundMask] = False

	return mat


def displayMask(hash, binarize=True):
	"""
		Displays the cilia mask against the given hash value.
	"""
	mask = readMask(hash, binarize)
	if binarize: im = Image.fromarray(mask * 255)
	else: im = Image.fromarray(mask * MASK_STRETCHING_CONSTANT)
	im.show()


def displayHeatMap(mat):
	"""
		Dispalys the heat map for the given matrix.
	"""
	matplot.imshow(mat, cmap='hot')
	matplot.show()


def readLines(filepath):
	"""
		Reads and returns the lines of the given file as a list.
	"""
	lines = []
	with open(filepath, 'r') as infile:
		for line in infile:
			lines.append(line.strip())

	return lines


def getVideoFramesDirectory(hash):
	"""
		Returns the video frames directory for the given hash.
	"""
	dir =  os.path.join(FRAMES_PATH, hash)
	if not os.path.isdir(dir): raise BadHashError("No frame directory found against the hash: " + hash)
	else: return dir


def mean(collection):
	"""
		Mean for a numeric collection
	"""
	return sum(collection) / (len(collection) or 1)


def stretchAndSaveMasks(hashes):
	"""
		This method stretches the contrast for the masks by rescaling them to 0-255 grayscale making the white regions in the masks cilia cells.
	"""
	# Read each mask and hash
	for hash in hashes:
		mask = readMask(hash, binarize=False)
		result = mask * MASK_STRETCHING_CONSTANT
		imsave(os.path.join(LIT_MASKS_PATH, hash + ".png"), result)


if __name__ == '__main__':
	# Quick testing etc.
	hashes = readLines(TRAIN_FILE)
	stretchAndSaveMasks(hashes)




