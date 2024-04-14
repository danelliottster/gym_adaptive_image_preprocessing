import torchvision
import torch
from torchvision.transforms import ToTensor, Pad , CenterCrop , RandomAffine, ColorJitter
from torchvision.transforms import functional as F
from torch.utils.data import Subset
import math

MNIST_ORIG_SIZE = 28
MIN_IMG_SIZE = 36

def create_corruption_dict( rotation_angle=0.0 , translation_x=0 , translation_y=0 , brightness_factor=0.0 ) :
    """
    Create a dictionary of corruptions to apply to the MNIST images.
    """
    return { "rotation" : { "angle" : rotation_angle } , 
             "translation" : { "x" : translation_x , "y" : translation_y } ,
             "brightness" : { "factor" : brightness_factor } }

def compute_padding_size( selected_corruptions ) :
    """
    Compute the minimum image size based upon the corruptions and padding.
    """
    min_img_size = MNIST_ORIG_SIZE + max( selected_corruptions["translation"]["x"] , selected_corruptions["translation"]["y"] )
    if "rotation" in selected_corruptions :
        min_img_size = max( min_img_size , math.ceil( MNIST_ORIG_SIZE * math.sqrt(2) ) )

    return min_img_size - MNIST_ORIG_SIZE

class MNISTCorrupted:

    def __init__( self , selected_corruptions , train_data_fraction = 1.0 , download_path='/mnt/c/Users/daniel.elliott/Downloads/mnist_tmp' ) :

        # the list of corruptions to use with this environment.
        self._selected_corruptions = selected_corruptions
        # the amount of padding to add to the image
        self._img_pad_size = compute_padding_size( self._selected_corruptions )
        # the size of the image after corruption and padding
        self._img_size = MNIST_ORIG_SIZE + self._img_pad_size
        # the number of training images
        self._N_train = None
        # the indicies of the training data set
        self._train_indices = None
        # the indicies of the unused training data set
        self._unused_train_indices = None
        # data loaders
        self._loader_train = None
        self._loader_train_unused = None
        self._loader_test = None

        # define the padding transform so we can use it later
        self._pad_func = Pad( self._img_pad_size , fill=0 )

        # load the MNIST data set
        self._mnist = torchvision.datasets.MNIST( root=download_path , train=True , download=True , transform=lambda x: ToTensor()(self._pad_func(x)) )
        N_mnist = len( self._mnist )

        # set the number of training images and determine indices
        self._N_train = int( train_data_fraction * N_mnist )
        self._train_indices = list( range( self._N_train ) )
        self._unused_train_indices = list( set( range( N_mnist ) ) - set( self._train_indices ) )
        self._N_train_unused = len( self._unused_train_indices )

        # specify the training data set (sometimes we want to use a subset of the training data)
        mnist_train_subset = Subset( self._mnist , self._train_indices)
        mnist_train_subset_unused = Subset( self._mnist , self._unused_train_indices)

        # load into a dataloader
        self._loader_train = torch.utils.data.DataLoader( mnist_train_subset , batch_size=128 , shuffle=True , num_workers=0 )
        if self._N_train_unused > 0 :
            self._loader_train_unused = torch.utils.data.DataLoader( mnist_train_subset_unused , batch_size=1 , shuffle=True , num_workers=0 )

        # specify the test data set
        mnist_test = torchvision.datasets.MNIST( root=download_path , train=False , download=True , transform=lambda x: ToTensor()(self._pad_func(x)) )
        self._loader_test = torch.utils.data.DataLoader( mnist_test , batch_size=1 , shuffle=False , num_workers=0 )

        # create the corruption functions
        self._affine_corruptor , self._brightness_corruptor = self.create_corruptor( rotation_angle=self._selected_corruptions["rotation"]["angle"] if "rotation" in self._selected_corruptions else 0 ,
                                                                                    translation_x=self._selected_corruptions["translation"]["x"] if "translation" in self._selected_corruptions else 0 ,
                                                                                    translation_y=self._selected_corruptions["translation"]["y"] if "translation" in self._selected_corruptions else 0 ,
                                                                                    brightness_factor=self._selected_corruptions["brightness"]["factor"] if "brightness" in self._selected_corruptions else 0 )
        
        # we'll use this to crop the image back to the original size
        self._cropper = CenterCrop( size=MNIST_ORIG_SIZE )

    def create_corruptor( self , rotation_angle=0.0 , translation_x=0 , translation_y=0 , brightness_factor=0.0 ) :
        """
        Create a corruptor function using RandomAffine and ColorJitter.
        """

        return RandomAffine( degrees=rotation_angle , translate=( translation_x , translation_y ) ) , ColorJitter( brightness=brightness_factor )


    def corrupt( self , image ) :
        """
        Corrupt an image.
        """
        # corrupt the image
        image = self._affine_corruptor( image )
        image = self._brightness_corruptor( image )
        # crop the image back to the original size
        image = self._cropper( image )

        return image

    def next_train( self ) :
        """
        Get the next corrupted image from the training data set.
        """
        # get the next image
        image_original , label = next( iter( self._loader_train ) )
        # corrupt the image
        image_corrupted = self.corrupt( image_original )

        return image_corrupted , label , image_original
    
    def next_train_unused( self ) :
        """
        Get the next corrupted image from the unused training data set.
        """
        if self._loader_train_unused :
            raise Exception( "No unused training data." )
        # get the next image
        image_original , label = next( iter( self._loader_train_unused ) )
        # corrupt the image
        image_corrupted = self.corrupt( image_original )

        return image_corrupted , label , image_original
    
    def next_test( self ) :
        """
        Get the next corrupted image from the unused training data set.
        """
        # get the next image
        image_original , label = next( iter( self._loader_test ) )
        # corrupt the image
        image_corrupted = self.corrupt( image_original )

        return image_corrupted , label , image_original