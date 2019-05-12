import cv2
import numpy as np

def HistogramEqualization(image):
	for channel in range(0, 2):
		image[:,:,channel] = cv2.equalizeHist(image[:,:,channel])
	return image

def unsharpEnhancement(image):
	unsharp_image = np.zeros(shape=image.shape)
	gaussian_3 = cv2.GaussianBlur(image, (9, 9), 10.0)
	unsharp_image = cv2.addWeighted(image, 1.5, gaussian_3, -0.5, 0, image)
	return unsharp_image

def Binarization(image):
	retval, binarized_img = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
	return binarized_img

def Smoothening(image, filter):
	height, width = image.shape
	windowSize = filter.shape[1]
	padSize = windowSize // 2		# Integer Division

	newImage = np.zeros(image.shape)
	filterSum = np.sum(filter)
	image = np.pad(image, (padSize, padSize), 'constant', constant_values=(0) )

	for x in range(padSize, height - padSize):
		for y in range(padSize, width - padSize):
			pixels = image[x-padSize:x+padSize+1, y-padSize:y+padSize+1]
			newImage[x-padSize,y-padSize] = np.sum(np.multiply(pixels, filter/filterSum))
	return newImage

def medianSmoothing(image, windowSize):
	height, width = image.shape
	padSize = windowSize // 2		# Integer Division
	newImage = np.zeros(image.shape)
	image = np.pad(image, (padSize, padSize), 'constant', constant_values=(0) )

	for x in range(padSize, height - padSize):
		for y in range(padSize, width - padSize):
			pixels = image[x-padSize:x+padSize+1, y-padSize:y+padSize+1]
			pixels = np.sort(pixels, axis=None)
			newImage[x-padSize, y-padSize] = pixels[(windowSize**2) // 2]
	return newImage