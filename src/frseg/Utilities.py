"""
			Author: Narinder Singh				Project: Cilia Segmentation			Date: 27 Feb 2019
			Course: CSCI 8360 @ UGA				Semester: Spring 2019				Module: Utilities.py

Description: This module contains methods and classes that make life easier.
"""

import os
import numpy as np
from PIL import Image
from Config import DATA_FILES_PATH
from Config import CILIA_GRAYSCALE

MASKS_PATH = os.path.join(DATA_FILES_PATH, "masks/")

class UtilitiesError(Exception): pass
class BadHashError(UtilitiesError): pass

def isImageFile(fpath):
	"""
		Returns whether or not the given path or filename is for an image file. The method just checks for some popular formats.
	"""
	path, fname = os.path.split(fpath)
	if fname.endswith(("png", "jpeg", "gif", "tiff", "bmp")): return True
	else: return False


def invertMask(mask):
	"""
		Inverts a numpy binary mask.
	"""
	return mask == False


def readMask(hash, binarize=False):
	"""
		Reads the mask for the given hash and if binarize flag is set, makes the mask binary (1/0 : Cilia/Not-cilia)
	"""
	fpath = os.path.join(MASKS_PATH, hash + ".png")
	if not os.path.isfile(fpath): raise BadHashError("Hash: " + hash + " does not exist OR does not have a mask against it.")

	img = Image.open(fpath)
	mat = np.asarray(img, np.int32)
	mat.setflags(write=1)
	
	if binarize:
	 	ciliaMask = mat == CILIA_GRAYSCALE
	 	backgroundMask = invertMask(ciliaMask)
	 	mat[ciliaMask] = 1
	 	mat[backgroundMask] = 0

	return mat


def displayMask(hash):
	"""
		Displays the cilia mask against the given hash value.
	"""
	mask = readMask(hash, binarize=True)
	im = Image.fromarray(mask * 255)
	im.show()


if __name__ == '__main__':
	# Quick testing
	displayMask("c581673f9684fd6952f38b6e0ae00a4e8eea6662d8635337da6bf067da0d9de7")



