import sys
sys.path.insert(0, '..')
from mnist_corrupted import MNISTCorrupted, create_corruption_dict
from classifiers import MnistClassifierSimpleNet

#######################################################
# load the corrupted MNIST data set

# no corruptions
corruptions = create_corruption_dict( )
mnist_corrupted = MNISTCorrupted( corruptions )

#######################################################
# train the classifier

classifier = MnistClassifierSimpleNet( mnist_corrupted._loader_train , mnist_corrupted._loader_test )
classifier.train()