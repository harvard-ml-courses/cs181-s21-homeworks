# CS 181, Spring 2020
# Homework 4

import numpy as np
import matplotlib.pyplot as plt

# This line loads the images for you. Don't change it!
pics = np.load("data/images.npy", allow_pickle=False)

print(pics.shape)

# You are welcome to change anything below this line. This is just an example of how your code may look.
# Keep in mind you may add more public methods for things like the visualization.
# Also, you must cluster all of the images in the provided dataset, so your code should be fast enough to do that.

class KMeans(object):
    # K is the K in KMeans
    def __init__(self, K):
        self.K = K

    # X is a (N x 28 x 28) array where 28x28 is the dimensions of each of the N images.
    def fit(self, X):
        pass

    # This should return the arrays for K images. Each image should represent the mean of each of the fitted clusters.
    def get_mean_images(self):
        pass

K = 10
KMeansClassifier = KMeans(K=10, useKMeansPP=False)
KMeansClassifier.fit(pics)

# This is how to plot an image. We ask that any images in your writeup be grayscale images, just as in this example.
plt.figure()
plt.imshow(pics[0].reshape(28,28), cmap='Greys_r')
plt.show()


class HAC(object):
	def __init__(self, linkage):
		self.linkage = linkage
