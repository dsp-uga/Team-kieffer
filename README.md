# CSCI 8360 - Cilia Sigmentation | Project II
## Team Keiffer


# Introduction
Cilia are microscopic hair like structure that protrude form cell bodies. This project aims to implement methods to segment cilia pixels from a cell body video (a series of images). Presently, this project implements just one method - a statistical approach to distinguish cilia pixels from the static background. To get a feel for the problem, it would be adviseable to explore the project dataset that's in the `files/` directory. The full problem definition can be accessed (here)[https://github.com/dsp-uga/sp19/blob/master/projects/p2/project2.pdf].

# Pre-requisites
This package relies on `Python Imaging Library (Pillow)`, `numpy`, and `matplotlib`, each of which is easily installable using pip - the python package manager as:
```
pip install pillow
pip install numpy
pip install matplotlib
```

# Project Directory
The project directory is structured as follows:
```
.
|---README.md										# Master README
|---CONTRIBUTORS.md								
|---files/											# Contains project dataset
      |---data/
		    |---archives							# Compressed dataset files
		          |---<hash>.tar					
		    |---frames								# Uncompressed video frames
				   |---<hash>/
				         |---<frame00xx>.png		
      |---masks/									# Masks highlighting cilia pixels for a subset of dataset
            |---<hash>.png
            |---lit/								# The same masks with rescaled greyscale values to highlight things 
                 |---<hash>.png
      |---train.txt								# File specifying the subset of dataset to be used for training
      |---test.txt									# Likewise for test
      |---small_train.txt							# Specification like above but relatively small
      |---small_test.txt							# Likewise
|---src/
     |---thresholding								# Takes a mean thresholding approach to segmentation
              |---Config.py						# File specifying where to find data etc.
              |---FStat.py						# Main module implementing threhsolding and mask generation
              |---Utilities.py					# Helper methods to make life easier
			   |---predictions						# Dir to save the predicted masks
					   |---<hash.png>				
					   |---p2.tar					# Submission
					   |---Lit/					# Rescaled masks making them human visible
			  	         |---<hash>.png			
	 		   |---visuals/						# Dir to save visuals in
				      |---README.md				# File explaining the visuals
				      |---Thresholding/			
							  |---<hash>.png
				      |---Variance
				             |---<hash>.png
```

# How to run
To generate the masks, navigate to the `src/thresholding/` directory and run as:
```
	python FStat.py
```
The masks will be saved to the `src/thresholding/predictions` directory. To know about the method, refer the project Wiki.
