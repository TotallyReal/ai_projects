# Learning machine learning

I use this project to actually implement all sorts of machine learning algorithms, and in particular in computer vision, 
to understand better the details behind them. Along the way I make notes to help me with this goal, and you are
welcome to read them as well:

## Computer vision

Beginning with a simple introduction:

0. Introduction to computer vision: [corner and edge detection](vision/notes/sift.ipynb)

One interesting problem in computer vision, is taking several 2D images and combine them together.

If the pictures are taken from the same position, we can glue them together to form a single "panoramic view", 
and if taken from different positions, we can try to use them to do some **3D reconstruction**. I wrote several 
notes with implementation about these interesting problems:

1. Images from a fixed position: [Creating panoramic views](vision/notes/panorama.ipynb)
2. Images from many positions: [3D reconstruction](vision/notes/3d_reconstruction.ipynb)
3. General multiple view geometry: [the math behind stitching together 2D images](vision/notes/points_of_view.md)
4. Locating the second camera position: [and why we need skew-symmetric times orthogonal decompositions](vision/notes/skew_symmetric_orthogonal.md)
5. Sift comparisons: While I didn't write anything about it, I do have several examples of image
   comparisons using my implementation [here](vision/images/sift_comparisons/sift_compare.md).


## General machine learning

1. Studying Neural Networks: Starting with the simplest one - [MNIST](mnist/mnist.md)

2. Dimension reduction using [principal component analysis](fashion_mnist/pca.md)



Found it interesting and have some feedback? Want me to add another explanation to the list? If so, let me know.

---

**My homepage**: [https://prove-me-wrong.com/](https://prove-me-wrong.com/)

**Contact**:	 [totallyRealField@gmail.com](mailto:totallyRealField@gmail.com)

**Ofir David**

