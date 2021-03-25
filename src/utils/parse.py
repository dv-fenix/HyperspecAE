import configargparse as cfargparse
import os

import torch
import warnings


class ArgumentParser(cfargparse.ArgumentParser):
    def __init__(
            self,
            config_file_parser_class=cfargparse.YAMLConfigFileParser,
            formatter_class=cfargparse.ArgumentDefaultsHelpFormatter,
            **kwargs):
        super(ArgumentParser, self).__init__(
            config_file_parser_class=config_file_parser_class,
            formatter_class=formatter_class,
            **kwargs)

    @classmethod
    def defaults(cls, *args):
        """Get default arguments added to a parser by all ``*args``."""
        dummy_parser = cls()
        for callback in args:
            callback(dummy_parser)
        defaults = dummy_parser.parse_known_args([])[0]
        return defaults
        
    @classmethod
    def validate_model_opts(cls, model_opt):
        assert model_opt.encoder_type in ["deep", "shallow"], \
            "Unsupported encoder type %s" % model_opt.encoder_type
            
        assert model_opt.soft_threshold in ["SReLU", "SLReLU"], \
            "Unsupported thresholding operation %s" % model_opt.soft_threshold
            
        assert model_opt.soft_threshold != 'SLReLU', \
            "SLReLU is currently unsupported." 
            
        assert model_opt.activation in ['ReLU', 'Leaky-ReLU', 'Sigmoid'], \
            "Unsupported activation function %s" % model_opt.activation
            
        if model_opt.encoder_type == 'shallow':
            if model_opt.activation != None:
                print("Activation is not required for a shallow autoencoder")
    
    @classmethod
    def validate_train_opts(cls, train_opt):
        assert train_opt.num_bands > 1, \
            "HSI has to have a minimum of two spectral bands."
            
        assert train_opt.end_members > 0, \
            "Number of end-members must be positive."
            
        assert train_opt.batch_size > 0, \
            "Batch size must be positive."
            
        assert train_opt.learning_rate > 0, \
            "Learning rate must be positive."
            
        assert train_opt.epochs > 0, \
            "Number of epochs must be positive."
            
        assert train_opt.gaussian_dropout > 0, \
            "Mean of applied noise must be positive."
            
        assert train_opt.threshold > 0, \
            "Threshold must be positive."
            
        assert train_opt.objective in ["MSE", "SAD","SID"], \
            "Unsupported objective function %s" % train_opt.objective
            
        if train_opt.objective == "SID":
            warnings.warn('Spectral Information Divergence may be unstable.')
            
        if train_opt.end_members >= train_opt.num_bands:
            raise AssertionError(
            "Number of end members to be extracted can't be more \
            than the number of existing spectral signatures.")
        
        if train_opt.save_checkpt > train_opt.epochs:
            raise AssertionError(
            "Checkpoint should lie within the number of training iterations.")
        
    @classmethod
    def validate_extract_opts(cls, extract_opt):
        assert extract_opt.num_bands > 1, \
            "HSI has to have a minimum of two spectral bands."
            
        assert extract_opt.end_members > 0, \
            "Number of end-members must be positive."
            
        assert extract_opt.batch_size > 0, \
            "Batch size must be positive."
            
        assert extract_opt.learning_rate > 0, \
            "Learning rate must be positive."
            
        assert extract_opt.epochs > 0, \
            "Number of epochs must be positive."
            
        assert extract_opt.gaussian_dropout > 0, \
            "Mean of applied noise must be positive."
            
        assert extract_opt.threshold > 0, \
            "Threshold must be positive."
            
        if extract_opt.end_members >= extract_opt.num_bands:
            raise AssertionError(
            "Number of end members to be extracted can't be more \
            than the number of existing spectral signatures.")
            
            
        