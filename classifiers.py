import numpy as np
import torchvision
from torchvision.transforms import ToTensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from abc import ABC , abstractmethod
import matplotlib.pyplot as plt


class MnistClassifier( ABC ) :

    def __init__( self ) :

        # load the MNIST training data set
        self._mnist_train = torchvision.datasets.MNIST( root='/mnt/c/Users/daniel.elliott/Downloads/mnist_tmp' , train=True , download=True , transform=ToTensor() )
        self._train_loader = torch.utils.data.DataLoader( self._mnist_train , batch_size=1 , shuffle=True , num_workers=0 )
        # load the MNIST test data set
        self._mnist_test = torchvision.datasets.MNIST( root='/mnt/c/Users/daniel.elliott/Downloads/mnist_tmp' , train=False , download=True , transform=ToTensor() )
        self._test_loader = torch.utils.data.DataLoader( self._mnist_test , batch_size=1 , shuffle=True , num_workers=0 )


    @abstractmethod
    def train( self ) :
        """
        A method which trains the classifier on the MNIST data set.
        """
        pass

    @abstractmethod
    def classify( self , image_in ) :
        """ 
        A method which takes in an image and returns the inferred probability for each class.
        Args:
            image_in : a 1xMxM MNIST image as a numpy ndarray M is the image size (square).
        Returns:
            The probability of membership in each class.
        """ 
        pass

class MnistClassifierCNN( MnistClassifier ) :
    """
    A CNN for classifying MNIST images in PyTorch.
    Will not use softmax because we want to force the agent to tune the classifier.
    """

    class MnistCNN( nn.Module ) :
            
        def __init__( self ) :

            super().__init__()

            self._conv1 = nn.Conv2d( 1 , 10 , kernel_size=5 )
            self._conv2 = nn.Conv2d( 10 , 20 , kernel_size=5 )
            self._conv2_drop = nn.Dropout2d()
            self._fc1 = nn.Linear( 320 , 50 )
            self._fc2 = nn.Linear( 50 , 10 )

        def forward( self , x ) :

            x = F.relu( F.max_pool2d( self._conv1( x ) , 2 ) )
            x = F.relu( F.max_pool2d( self._conv2_drop( self._conv2( x ) ) , 2 ) )
            x = x.view( -1 , 320 )
            x = F.relu( self._fc1( x ) )
            x = F.dropout( x , training=self.training )
            x = self._fc2( x )
            

    def __init__( self ) : 

        super().__init__()
        # the classifier
        self._classifier = self.MnistCNN()

    def train( self ) :
            
        # define the loss function
        loss_fn = nn.CrossEntropyLoss()
        # define the optimizer
        optimizer = optim.SGD( self._classifier.parameters() , lr=0.01 , momentum=0.5 )

        # train the classifier
        for epoch in range( 10 ) :
            for batch_idx , ( data , target ) in enumerate( self._train_loader ) :
                optimizer.zero_grad()
                output = self._classifier( data )
                loss = loss_fn( output , target )
                loss.backward()
                optimizer.step()

    def test( self ) :

        test_loss = 0
        correct = 0
        with torch.no_grad() :
            for data , target in self._test_loader :
                output = self._classifier( data )
                test_loss += F.nll_loss( output , target , reduction='sum' ).item()
                pred = output.argmax( dim=1 , keepdim=True )
                correct += pred.eq( target.view_as( pred ) ).sum().item()

        test_loss /= len( self._test_loader.dataset )
        print( '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss , correct , len( self._test_loader.dataset ) ,
            100. * correct / len( self._test_loader.dataset ) ) )
        
    def classify( self , image_in ) :

        # convert the image to a PyTorch tensor
        image_in = torch.from_numpy( image_in ).float()
        # add a batch dimension
        image_in = image_in.unsqueeze( 0 )
        # run the classifier
        output = self._classifier( image_in )
        # convert the output to a numpy array
        output = output.detach().numpy()
        # return the output of the CNN
        return output
    
class MnistClassifierAutoCorr( MnistClassifier ) :
    """
    A class which implements a classifier for MNIST images using the autocorrelation method.
    Each class will be represented by a mean image.
    Classification will be done using autocorrelation against each mean image.
    Should be a good test for the agent with a more informative reward during training.
    """

    def __init__( self ) :

        super().__init__()

    def plot_mean_images( self ) :

        fig , ax = plt.subplots( 2 , 5 )
        for i in range( 10 ) :
            ax[ i // 5 , i % 5 ].imshow( self._mean_images[ i , : , : ] , cmap='gray' )
        plt.show()

    def train( self ) :

        # compute the mean image for each class
        self._mean_images = np.zeros( ( 10 , 28 , 28 ) )
        self._num_images = np.zeros( 10 )
        for image , label in self._train_loader :
            image = image.squeeze().numpy()
            label = label.squeeze().numpy()
            self._mean_images[ label , : , : ] += image
            self._num_images[ label ] += 1
        for i in range( 10 ) :
            self._mean_images[ i , : , : ] /= self._num_images[ i ]

    def test( self ) :

        # compute the accuracy of the classifier
        num_correct = 0
        num_total = 0
        for idx , (image , label) in enumerate( self._test_loader ):
            image = image.squeeze().numpy()
            label = label.squeeze().numpy()
            autocorr = self.classify( image )
            print( f'Testing image {idx} of {len( self._test_loader )} with autocorr of {autocorr[label]}' )
            if np.argmax( autocorr ) == label :
                num_correct += 1
            num_total += 1
        print( 'Accuracy: ' + str( num_correct / num_total ) )

    def classify( self , image_in ) :

        # compute the autocorrelation of the input image against each mean image
        match_score = np.zeros( 10 )
        for i in range( 10 ) :
            autocorr = np.sum( np.multiply( image_in , self._mean_images[ i , : , : ] ) )
            
            # subtract the part of the input image that is missing from the mean image
            binary_mean_img = np.zeros( ( 28 , 28 ) )
            binary_mean_img[ self._mean_images[ i , : , : ] > 0.1 ] = 1
            missing_from_image_in = image_in - binary_mean_img
            missing_from_image_in[ missing_from_image_in < 0 ] = 0
            
            # some reward shaping
            match_score[ i ] = 2 * autocorr - np.sum( missing_from_image_in )

        # return the match score
        return match_score