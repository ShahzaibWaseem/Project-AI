# Ship Detection using Aerial images
This is a semester project for the course "Artificial Intelligence".

## Description
Ship detection from a dataset consisting of aerial images. Image processing techniques like Binarization, Histogram Equalization, Unsharping and Smoothing (using different kernels and median Smoothing) were employed which yielded better results and the processed images were tested on UNET and VGG models.

## Loss Function
UNET's Loss Function


![Loss Function](https://github.com/ShahzaibWaseem/Project-AI/blob/master/Images/10%20Epochs.png)


|Custom Model Without DIP|Custom Model With DIP|
|--|--|
|![without IP](https://github.com/ShahzaibWaseem/Project-AI/blob/master/Images/10%20Epochs%20%28Custom%20Model%29.png)|![with IP](https://github.com/ShahzaibWaseem/Project-AI/blob/master/Images/10%20Epochs%20%28Custom%20Model%29%20%28with%20DIP%29.png)|