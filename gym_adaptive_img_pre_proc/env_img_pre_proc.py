import gym
from gym import spaces
import numpy as np
import torchvision
import torch
from torchvision.transforms import ToTensor, Pad , CenterCrop , RandomAffine, ColorJitter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from abc import ABC , abstractmethod
import matplotlib.pyplot as plt
import random

CORRUPTIONS = [ 'translation' , 'rotation' , 'brightness' ]
ACTION_COUNTS = { 'translation':2 , 'rotation':1 , 'brightness':1 }
ACTION_SCALES = { 'translation':1 , 'rotation':1 , 'brightness':0.1 }
ACTION_INIT_VALS = { 'translation':0 , 'rotation':0 , 'brightness':1 }
ACTION_RANGE = { 'translation':[-40,40] , 'rotation':[-45,45] , 'brightness':[0.4,5] }
REWARD_SCALAR = 100
MNIST_ORIG_SIZE = 28
MIN_IMG_SIZE = 36

def create_empty_cumulative_actions() :
    """
    Create an empty dictionary of cumulative actions
    Returns:
        the empty dictionary
    """
    cumulative_actions = {}
    for action_name , action_dim in ACTION_COUNTS.items() :
        init_val = ACTION_INIT_VALS[ action_name ]
        cumulative_actions[ action_name ] = np.zeros( action_dim ) + init_val
    return cumulative_actions


class MnistCorrupted( gym.Env ) :
    """
    A subclass of the openai gym environment class.
    The object of this environment is to undo the transformations which are applied to the original image. 
    The environment is initialized with a classifier which is used to determine the reward for each action.
    A pad of zeros is added to the image to allow for translation and to make sure the image is large enough for the default-sized CNN.
    """

    def __init__( self , classifier , img_pad_size = 10 , max_duration = 50 , selected_corruptions=["translation"] , test_env=False , num_test_imgs=25 ) :
        """ Initialize the environment
        
        TODO:
            - modify reward function to penalize actions which do not improve the classification
            - add more corruptions
            - add a null action
        Args:
            classifier: the classifier to use for the reward function.  Must be a subclass of MnistClassifier.
            img_pad_size: the number of zeros to pad the image with in each direction.
            max_duration : maximum number of time step for each trial
            selected_corruptions: a list of of strings.  Each string is a corruption name which must be found in CORRUPTIONS.
            test_env: if True, the environment will use the test set instead of the training set.
            num_test_imgs: the number of test images to use when evaluating performance. Only used if test_env is True.
        """

        super().__init__()

        # the data loader used when this is a test environment.  Will be None if this is not a test environment.
        self._test_loader = None
        # the list of corruptions to use with this environment.
        self._selected_corruptions = None
        # the amount of padding to add to the image
        self._img_pad_size = None
        # the size of the image
        self._img_size = None
        # the number of actions in the action space
        self._N_actions = None
        # the action space
        self.action_space = None
        # the observation space
        self.observation_space = None
        # the MNIST training/testing data sets
        self._mnist_train = None
        self._mnist_test = None
        

        # handle the provided corruptions
        self._selected_corruptions = selected_corruptions
        assert np.all( [ corruption in CORRUPTIONS for corruption in self._selected_corruptions ] ) , "One more more supplied corruption types are invalid."

        # compute observation space size
        self._img_pad_size = img_pad_size
        self._img_size = MNIST_ORIG_SIZE + 2 * self._img_pad_size
        if self._img_size < MIN_IMG_SIZE :
            #throw exception
            raise Exception( "The image size is too small.  The image size must be at least 36 pixels." )

        # setup action and observation spaces
        self._N_actions = sum( [ ACTION_COUNTS[ corruption ] for corruption in CORRUPTIONS ] ) * 2 # for positive and negative directon for each action type
        self.action_space = spaces.Discrete( self._N_actions )
        self.observation_space = spaces.Box( low=0 , high=255 , shape=( self._img_size , self._img_size , 1 ) , dtype=np.uint8 )

        # load mnist dataset from torchvision
        # load into a dataloader
        self._mnist_train = torchvision.datasets.MNIST( root='/mnt/c/Users/daniel.elliott/Downloads/mnist_tmp' , train=True , download=True , transform=lambda x: ToTensor()(Pad( self._img_pad_size, fill=0 )(x)) )
        self._train_loader = torch.utils.data.DataLoader( self._mnist_train , batch_size=1 , shuffle=True , num_workers=0 )

        # create a loader for the test set
        self.num_test_imgs = num_test_imgs
        if test_env :
            self._mnist_test = self._mnist_train.data[ random.sample( range(60000) , num_test_imgs ) , : , : ]
            self._test_loader = torch.utils.data.DataLoader( self._mnist_test , batch_size=1 , shuffle=False , num_workers=0 )
        else :
            self._mnist_test = None
            self._test_loader = None

        # setup the state variables
        self._state_img_orig = None
        self._state_img_label = None
        self._state_img_classifier_label = None
        self._state_corruption_params = {} #currently unused
        self._state_img_corrupted = None
        self._state_cumulative_actions = create_empty_cumulative_actions()
        self._state_num_steps = 0
        self._state_last_action_noop = False #currently unused
        self._max_duration = max_duration

        # the classifier to use for the reward function
        self._classifier = classifier

        # create corruption functions
        self._affine_corruptor = RandomAffine( degrees=45 if "rotation" in self._selected_corruptions else 0 , 
                                                translate = ( 0.40 , 0.30 ) if "translation" in self._selected_corruptions else ( 0 , 0 ) ,
                                                scale=None , shear=None )
        self._brightness_corruptor = ColorJitter( brightness = 0.5 if "brightness" in self._selected_corruptions else 0 )
        
        # we'll use this to crop the image back to the original size
        self._cropper = CenterCrop( size=MNIST_ORIG_SIZE )

    def check_if_observation_is_blank( self , observation ) :

        if np.max( observation ) == 0 :
            return True
        else :
            return False
    
    def is_done( self , observation , verbose=0 ) :
        """
        Check if the episode is done.  Episode is done if number of steps exceeds max duration or if all pixels are zero.
        Args:
            observation: the current observation as a numpy ndarray
            verbose: if > 0, print messages to the console
        Returns:
            True if the episode is done, False otherwise
        """

        if self._state_num_steps >= self._max_duration :
            if verbose > 0 :
                print( "Max duration reached.  Terminating trial." )
            return True
        
        if self.check_if_observation_is_blank( observation ) :
            if verbose > 0 :
                print( "All pixels are zero.  Terminating trial." )
            return True
        
        else :
            return False

    def step( self , action , verbose=False ) :

        self._state_num_steps += 1

        # unpack the action
        unpacked_action = self.unpack_action( action )

        # accumulate the action
        self.accumulate_action( unpacked_action )

        # apply the cumulative actions to the original image
        observation = self.apply_cumulative_actions( )

        # calculate the reward
        reward = self.reward_function( observation.squeeze().numpy() , verbose=verbose )

        # check if the episode is over
        done = self.is_done( observation=observation.squeeze().numpy() , verbose=verbose )

        # return the observation, reward, done, truncated, and info
        return observation.numpy().astype(np.uint8) , reward , done , {}
    
    def reset( self ) :

        # reset the cumulative actions
        self._state_cumulative_actions = create_empty_cumulative_actions()

        # reset the step counter
        self._state_num_steps = 0

        # get the next image from the dataset
        if self._test_loader is None :
            self._state_img_orig , self._state_img_label = next( iter( self._train_loader ) )
            self._state_img_label = self._state_img_label.squeeze().numpy()
        else :
            self._state_img_orig = next( iter( self._test_loader ) )
            self._state_img_label = None

        self._state_img_orig = self._state_img_orig.squeeze().numpy()

        # determine the label of the original image using the classifier
        tmp_img = self._cropper( torch.from_numpy( np.expand_dims( self._state_img_orig , axis=2 ) ) ).squeeze().numpy()
        self._state_img_classifier_label = np.argmax( self._classifier.classify( tmp_img ) )

        # corrupt the image
        self._state_img_corrupted = self._affine_corruptor( torch.from_numpy( np.expand_dims( self._state_img_orig , axis=0 ) ) )
        self._state_img_corrupted = self._brightness_corruptor( self._state_img_corrupted ).squeeze().numpy()

        # return the observation
        return np.expand_dims( self._state_img_corrupted , axis=2 ).astype( np.uint8 )
        
    def partial_reset( self ) :

        # reset the cumulative actions
        self._state_cumulative_actions = create_empty_cumulative_actions()

        # reset the step counter
        self._state_num_steps = 0

        # return the observation
        return np.expand_dims( self._state_img_corrupted , axis=2 ).astype( np.uint8 )

    def close( self ) :
        pass

    def count_action_space( self ) :

        return [ ACTION_COUNTS[ corruption ] for corruption in self._selected_corruptions ]

    def unpack_action( self , action_idx ) :
        """
        Unpack the action from a scalar to a dictionary of actions.
        The action index is assumed to be in the order of the selected corruptions and each action has a value in {-1,+1}.  THe negatively-valued action will come first.
        """
        idx = 0
        unpacked_action = {}
        for action_name in self._selected_corruptions :
            action_scale = ACTION_SCALES[ action_name ]
            tmp_size = len( self._state_cumulative_actions[ action_name ] )
            unpacked_action[ action_name ] = np.zeros( tmp_size )
            for tmp_idx in range( tmp_size ) :
                
                if idx == action_idx :
                    unpacked_action[ action_name ][ tmp_idx ] = -1 * action_scale
                idx += 1

                if idx == action_idx :
                    unpacked_action[ action_name ][ tmp_idx ] = 1 * action_scale
                idx += 1

        return unpacked_action

    def accumulate_action ( self , unpacked_actions ) :

        for action_name , action_value in unpacked_actions.items() :
            min_val = ACTION_RANGE[ action_name ][0]
            max_val = ACTION_RANGE[ action_name ][1]
            self._state_cumulative_actions[ action_name ] += action_value
            self._state_cumulative_actions[ action_name ] = np.clip( self._state_cumulative_actions[ action_name ] , min_val , max_val )

    def reward_function( self , observed_image , verbose=False ) :
        """
        Reward function for the environment.
        TOOD: 
            Add a penalty for each action taken.
            Use the correct classification in the reward computation.
        Args:
            observed_image: The image that the agent has observed.  A numpy ndarray.
        Returns:
            The reward for the current state.
        """
        reward = 0.0

        # what does the classifier think the image is?
        tmp_img = self._cropper( torch.from_numpy( np.rollaxis( np.expand_dims( observed_image , axis=2 ) , axis=2 , start=0 ) ) ).squeeze().numpy()
        class_inferences = self._classifier.classify( tmp_img )

        if ( self.check_if_observation_is_blank( observed_image ) ) :
            reward -= 100.0

        if verbose :
            print( f"Class inferences: {class_inferences}" )

        reward += np.max( class_inferences )

        return reward

    def apply_cumulative_actions( self ) :
        """ 
        Apply the cumulative actions to the original image.
        First, apply the translation.
        Second, apply the rotation.
        Third, apply the scaling.
        Fourth, apply the shearing.
        Returns the resulting observation: MxMx1 numpy array.
        """

        # apply translation
        observation = torchvision.transforms.functional.affine( torch.from_numpy( np.expand_dims( self._state_img_corrupted , axis=0 ) ), 
                                                               angle = 0 , 
                                                               translate=self._state_cumulative_actions[ "translation" ].tolist() ,
                                                               scale = 1 , shear=0 )

        # apply rotation
        observation = torchvision.transforms.functional.affine( observation , 
                                                               angle = self._state_cumulative_actions[ "rotation" ][0] , 
                                                               translate=(0,0) ,
                                                               scale = 1 , shear=0 )
        
        # apply brightness
        observation = torchvision.transforms.functional.adjust_brightness( observation , self._state_cumulative_actions[ "brightness" ][0] )

        # # apply scaling
        # observation = torchvision.transforms.functional.affine( observation , 
        #                                                        angle = 0 , 
        #                                                        translate=(0,0) ,
        #                                                        scale = self._state_cumulative_actions[ "scaling" ][0] , shear=0 )

        # # apply shearing
        # observation = torchvision.transforms.functional.affine( observation , 
        #                                                        angle = 0 , 
        #                                                        translate=(0,0) ,
        #                                                        scale = 1 , shear=self._state_cumulative_actions[ "shearing" ][0] )

        observation = observation.squeeze().unsqueeze(2)

        return observation
    
    def render( self , mode='human' , fig_in=None ) :
        """
        Draw the current state of the environment.
        If fig_in is None, create a new figure. Otherwise, update the figure.
        Figure consists of three subplots:
            1. Original image
            2. Current image
            3. Corrupted image
        """
        
        if mode != 'human' :
            raise NotImplementedError( "Only human mode is supported." )

        observation = self.apply_cumulative_actions( ).squeeze().numpy()

        if not fig_in :

            fig = plt.figure()
            ax_orig = fig.add_subplot( 1 , 3 , 1 )
            ax_orig.set_title( "Original" )
            im_orig = ax_orig.imshow( self._state_img_orig , cmap='gray' )
            ax_current = fig.add_subplot( 1 , 3 , 2 )
            ax_current.set_title( "Observed" )
            im_current = ax_current.imshow( observation , cmap='gray' )
            ax_corr = fig.add_subplot( 1 , 3 , 3 )
            ax_corr.set_title( "Corrupted" )
            im_corr = ax_corr.imshow( self._state_img_corrupted , cmap='gray' )
            fig.show()

        else :

            ax_orig = fig_in.axes[ 0 ]
            ax_orig.images[ 0 ].set_data( self._state_img_orig )
            ax_current = fig_in.axes[ 1 ]
            ax_current.images[ 0 ].set_data( np.squeeze( observation ) )
            ax_corr = fig_in.axes[ 2 ]
            ax_corr.images[ 0 ].set_data( self._state_img_corrupted )
            fig_in.canvas.draw()
            fig_in.canvas.flush_events()
            fig = fig_in

        return fig

