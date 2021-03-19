# Starter code for use with autograder.
import numpy as np
import matplotlib.pyplot as plt


def get_cumul_var(mnist_pics,
                  num_leading_components=500):

    """
    Perform PCA on mnist_pics and return cumulative fraction of variance
    explained by the leading k components.

    Returns:
        A (num_leading_components, ) numpy array where the ith component
        contains the cumulative fraction (between 0 and 1) of variance explained
        by the leading i components.

    Args:

        mnist_pics, (N x D) numpy array:
            Array containing MNIST images.  To pass the test case written in
            T5_P2_Autograder.py, mnist_pics must be a 2D (N x D) numpy array,
            where N is the number of examples, and D is the dimensionality of
            each example.

        num_leading_components, int:
            The variable representing k, the number of PCA components to use.
    """

    # TODO: compute PCA on input mnist_pics

    # TODO: return a (num_leading_components, ) numpy array with the cumulative
    # fraction of variance for the leading k components
    ret = np.zeros(num_leading_components)

    return ret

# Load MNIST.
mnist_pics = np.load("data/images.npy")

# Reshape mnist_pics to be a 2D numpy array.
num_images, height, width = mnist_pics.shape
mnist_pics = np.reshape(mnist_pics, newshape=(num_images, height * width))

num_leading_components = 500

cum_var = get_cumul_var(
    mnist_pics=mnist_pics,
    num_leading_components=num_leading_components)

# Example of how to plot an image.
plt.figure()
plt.imshow(mnist_pics[0].reshape(28,28), cmap='Greys_r')
plt.show()


