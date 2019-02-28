"""
			Author: Narinder Singh				Project: Cilia Segmentation			Date: 27 Feb 2019
			Course: CSCI 8360 @ UGA				Semester: Spring 2019				Module: Utilities.py

Description: This module contains methods and classes that make life easier.
"""

import os

def isImageFile(fpath):
	"""
		Returns whether or not the given path or filename is for an image file. The method just checks for some popular formats.
	"""
	path, fname = os.path.split(fpath)
	if fname.endswith("png", "jpeg", "gif", "tiff", "bmp"): return True
	else: return False
