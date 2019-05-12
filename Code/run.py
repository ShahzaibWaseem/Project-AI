import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
from model import buildModel
import imageProcessing

from keras.models import load_model
from keras.optimizers import *
from keras import backend as K
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

DATA_DIR = "../data/"
IMAGE_SIZE = 768
BATCH_SIZE = 4
VALIDATION_BATCH_SIZE = 2
NUM_EPOCHS = 10

#9c9d84825.jpg

def main():
	trainDataFile = pd.read_csv(DATA_DIR+'train_ship_segmentations_v2.csv')
	print("Training Data (csv) File Shape:\t", trainDataFile.shape)

	# Labeling NaN values
	trainDataFile['NaN'] = trainDataFile['EncodedPixels'].apply(isNaN)
	trainDataFile = trainDataFile.iloc[100000:]
	trainDataFile = trainDataFile.sort_values('NaN', ascending=False)
	print("\nNaN Value Count")
	print(trainDataFile['NaN'].value_counts())

	# Calculating Areas
	trainDataFile['area'] = trainDataFile['EncodedPixels'].apply(calculateArea)
	IsShip = trainDataFile[trainDataFile['area'] > 0]

	train_group = trainDataFile.groupby('ImageId').sum()
	print("\nGrouping Entries with same ImageId\nNaN Value Count")
	print(trainDataFile['NaN'].value_counts())

	train_group = train_group.reset_index()

	# Assigning Classes
	train_group['class'] = train_group['area'].apply(assignClasses)
	print("\nClasses of Ships")
	print(train_group['class'].value_counts())

	# Train, Validation Split
	trainingSet, validationSet = train_test_split(train_group, test_size=0.01, stratify=train_group['class'].tolist())
	print("\nTraining Set Shape:\t", trainingSet.shape[0])
	print("Validation Set Shape:\t", validationSet.shape[0])

	trainingSet_ship = trainingSet['ImageId'][trainingSet['NaN']==0].tolist()
	trainingSet_nan = trainingSet['ImageId'][trainingSet['NaN']==1].tolist()
	# Randomizing
	trainingSet_ship = random.sample(trainingSet_ship, len(trainingSet_ship))
	trainingSet_nan = random.sample(trainingSet_nan, len(trainingSet_nan))
	EQUALIZED_DATA = min(len(trainingSet_ship),len(trainingSet_nan))

	validationSet_ship = validationSet['ImageId'][validationSet['NaN']==0].tolist()
	validationSet_nan = validationSet['ImageId'][validationSet['NaN']==1].tolist()
	print("Training Set (Ships, Not Ships):\t", len(trainingSet_ship), len(trainingSet_nan))

	datagen = customGenerator(trainDataFile, trainingSet_ship, trainingSet_nan, batchSize=BATCH_SIZE, equalizedData=EQUALIZED_DATA)
	valgen = customGenerator(trainDataFile, validationSet_ship, validationSet_nan, batchSize=VALIDATION_BATCH_SIZE, equalizedData=EQUALIZED_DATA)

	validation_x, validation_y = next(valgen)

	model = buildModel()
	model.summary()

	model.compile(optimizer=Adam(1e-3, decay=0.0), metrics=['accuracy', f1], loss='mean_squared_error')

	model.save('ship_imageProcessing.h5')

	# model = load_model('ship.h5', custom_objects={"f1" : f1})
	# Training
	history = model.fit_generator(datagen,
		steps_per_epoch = 250,
		epochs = NUM_EPOCHS,
		verbose = 1,
		validation_data=(validation_x, validation_y))

	plt.subplot(2, 1, 1)
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('Model Accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='lower right')

	plt.subplot(2, 1, 2)
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model Loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper right')

	plt.tight_layout()
	plt.savefig('../Images/ModelPerformance.png')
	plt.show()

	predictions = model.predict(validation_x)
	# print("First Prediction: ", predictions[0])

	score = model.evaluate(validation_x, validation_y, verbose=1)
	print('Validation loss:', score[0])
	print('Validation accuracy:', score[1])

def isNaN(x):
	return 0 if (x == x) else 1

def calculateArea(EncodedPixels):
    EncodedPixels_list = [int(x) if x.isdigit() else x for x in str(EncodedPixels).split()]
    return 0 if (len(EncodedPixels_list) == 1) else np.sum(EncodedPixels_list[1::2])

def assignClasses(area):
    area = area / (IMAGE_SIZE * IMAGE_SIZE)
    if area == 0: return 0
    elif area < 0.005: return 1
    elif area < 0.015: return 2
    elif area < 0.025: return 3
    elif area < 0.035: return 4
    elif area < 0.045: return 5
    else: return 6

def encoding_masking(EncodedPixels_list, SHAPE):
    tmp_flat = np.zeros(SHAPE[0] * SHAPE[1])
    if len(EncodedPixels_list) == 1:
        mask = np.reshape(tmp_flat, SHAPE).T
    else:
        strt = EncodedPixels_list[::2]
        length = EncodedPixels_list[1::2]
        for i,v in zip(strt,length):
            tmp_flat[(int(i)-1):(int(i)-1)+int(v)] = 255
        mask = np.reshape(tmp_flat, SHAPE).T
    return mask

def customGenerator(trainDataFile, shipList, nanList, batchSize, equalizedData):
    trainingSet_nan_names = shipList[:equalizedData]
    trainingSet_ship_names = nanList[:equalizedData]
    k = 0
    while True:
        if k + batchSize // 2 >= equalizedData:
            k = 0
        batch_nan_names = trainingSet_nan_names[k : k + batchSize // 2]
        batch_ship_names = trainingSet_ship_names[k : k + batchSize // 2]
        batch_images = []
        batch_mask = []
        for name in batch_nan_names:
            image = cv2.imread(DATA_DIR + 'train_v2/' + name)
            # image = imageProcessing.Binarization(image)
            image = imageProcessing.unsharpEnhancement(image)
            image = imageProcessing.HistogramEqualization(image)
            batch_images.append(image)
            mask_list = trainDataFile['EncodedPixels'][trainDataFile['ImageId'] == name].tolist()
            oneMask = np.zeros((768, 768, 1))
            for mask_i in mask_list:
                EncodedPixels_list = str(mask_i).split()
                tempMask = encoding_masking(EncodedPixels_list, (768, 768))
                oneMask[:,:,0] += tempMask
            batch_mask.append(oneMask)
        for name in batch_ship_names:
            image = cv2.imread(DATA_DIR + 'train_v2/' + name)
            batch_images.append(image)
            mask_list = trainDataFile['EncodedPixels'][trainDataFile['ImageId'] == name].tolist()
            oneMask = np.zeros((768, 768, 1))
            for mask_i in mask_list:
                EncodedPixels_list = str(mask_i).split()
                tempMask = encoding_masking(EncodedPixels_list, (768, 768))
                oneMask[:,:,0] += tempMask
            batch_mask.append(oneMask)
        img = np.stack(batch_images, axis=0)
        mask = np.stack(batch_mask, axis=0)
        img = img / 255.0
        mask = mask / 255.0
        k += batchSize // 2
        yield img, mask

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

if __name__ == '__main__':
	main()