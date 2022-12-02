## Train

Usage: python train.py [path_to_checkpoint_directory] --cfg=[name_of_config_file]

## Resume Training

Usage: python train.py [path to checkpoint directory] --cfg=[name of config file] --rsm[path to latest model params OR path to resume_training.pth file]

## Eval

Usage: python eval.py [path_to_checkpoint_directory] --cfg=[name_of_config_file] --weights=[name_of_weight_file]

All hyperparameters can be adjusted via config file. The same config file should be used for train and eval.

