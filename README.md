An experimental work.

What if we can differentiate digits or non digits character from a model as a
result of learning 10 digits?

The model is based on Convolutional Neural Network classifying 10 digits MNIST
saved in ```model``` directory. The accuracy is around 98%.

### Generate notMNIST file
For testing, we are using notMNIST data from here http://yaroslavvb.blogspot
.com/2011/09/notmnist-dataset.html. There are 10k of images in
```data/notMNIST``` directory containing 1k images from each letter (A-J).
Because it is much faster to load the pickled data rather than reading it
from filesystems one by one, you can generate the pickled file by running:
```python utilities.py -d save_notmnist```. It will saved as ```data/notMNIST
.pkl```. The pickled file is not included in git since the size is ~100MB.
