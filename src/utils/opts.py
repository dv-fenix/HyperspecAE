from __future__ import print_function

import configargparse

def model_opts(parser):
    """
    These options are passed to the construction of the model.
    Be careful with these as they will be used during unmixing.
    """
    group = parser.add_argument_group('Model AE')
    group.add('--encoder_type', '-encoder_type', type=str, default='shallow',
    choices=['deep', 'shallow'],
    help="Allows the user to choose between two levels of encoder complexity."
         "Options are: [deep|shallow]")
         
    # SLReLU unavailable, add assert in main    
    group.add('--soft_threshold', '-soft_threshold', type=str, default='SReLU',
    choices=['SReLU', 'SLReLU'],
    help="Type of soft-thresholding for final layer of encoder"
         "Options are: [SReLU|SLReLU]")
         
    group.add('--activation', '-activation', type=str,
    choices=['ReLU', 'Leaky-ReLU', 'Sigmoid'],
    help="Activation function for hidden layers of encoder."
         "For shallow AE there won't be any activation. Options are:"
         "[ReLU|Leaky-ReLU|Sigmoid]")
         
def train_opts(parser):
    """
    These options are passed to the training of the model.
    Be careful with these as they will be used during unmixing.
    """
    group = parser.add_argument_group('General')
    group.add('--src_dir', '-src_dir', type=str, required=True,
    help="System path to the Samson directory.")
    
    group.add('--save_checkpt', '-save_checkpt', type=int, default=0,
    help="Number of epochs after which a check point of"
          "model parameters should be saved.")
          
    group.add('--save_dir', '-save_dir', type=str, default="../logs",
    help="System path to save model weights.")
    
    group.add('--train_from', '-train_from', type=str, default=None,
    help="Path to checkpoint file to continue training from.")
    
    group.add('--num_bands', '-num_bands', type=int, default=156,
    help="Number of spectral bands present in input image.")
    
    group.add('--end_members', '-end_members', type=int, default=3,
    help="Number of end-members to be extracted from HSI.")
    
    group = parser.add_argument_group('Hyperparameters')
    group.add('--batch_size', '-batch_size', type=int, default=20,
    help="Maximum batch size for training.")
    
    group.add('--learning_rate','-learning_rate', type=float, default=1e-3,
    help="Learning rate for training the network.")
    
    group.add('--epochs','-epochs', type=int, default=100,
    help="Number of iterations that the network should be trained for.")
    
    group.add('--gaussian_dropout', '-gaussian_dropout', type=float, default=1.0,
    help="Mean of multiplicative gaussain noise used for regularization.")
    
    group.add('--threshold', '-threshold', type=float, default=5.0,
    help="Defines the threshold for the soft-thresholding operation.")
    
    group.add('--objective', '-objective', type=str, default='MSE',
    choices=['MSE', 'SAD', 'SID'],
    help="Objective function used to train the Autoencoder."
         "Options are: [MSE|SAD|SID]")
    
    
         
         