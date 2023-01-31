#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to train the DTLN model in default settings. The folders for noisy and
clean files are expected to have the same number of files and the files to 
have the same name. The training procedure always saves the best weights of 
the model into the folder "./models_'runName'/". Also a log file of the 
training progress is written there. To change any parameters go to the 
"DTLN_model.py" file or use "modelTrainer.parameter = XY" in this file.
It is recommended to run the training on a GPU. The setup is optimized for the
DNS-Challenge data set. If you use a custom data set, just play around with
the parameters.

Please change the folder names before starting the training. 

Example call:
    $python run_training.py

Author: Nils L. Westhausen (nils.westhausen@uol.de)
Version: 13.05.2020

This code is licensed under the terms of the MIT-license.
"""

from DTLN_model import DTLN_model
import os

# use the GPU with idx 0
os.environ["CUDA_VISIBLE_DEVICES"]='1'
# activate this for some reproducibility
os.environ['TF_DETERMINISTIC_OPS'] = '1'
#os.environ["CUDA_VISIBLE_DEVICES"]='2'
# activate this for some reproducibility
# os.environ['TF_DETERMINISTIC_OPS'] = '3'


# path to folder containing the noisy or mixed audio training files
path_to_train_mix = '/data/projects/DNS-Challenge-master/ljl_200h_gap/train/noisy' #'/data/projects/v2/train/noisy'
# path to folder containing the clean/speech files for training
path_to_train_speech = '/data/projects/DNS-Challenge-master/ljl_200h_gap/train/clean' #'/data/projects/v2/train/clean'
# path to folder containing the noisy or mixed audio validation data
path_to_val_mix = '/data/projects/DNS-Challenge-master/ljl_200h_gap/val/noisy' #'/data/projects/v2/verify/noisy'
# path to folder containing the clean audio validation data
path_to_val_speech = '/data/projects/DNS-Challenge-master/ljl_200h_gap/val/clean' #'/data/projects/v2/verify/clean'

# name your training run
runName = 'DTLN_ns_B3_200h_norm'

# create instance of the DTLN model class
modelTrainer = DTLN_model()

# build the model
modelTrainer.build_DTLN_model_B3(norm_stft=True)
#modelTrainer.build_DTLN_model_stateful(norm_stft=True)

# compile it with optimizer and cost function for training
modelTrainer.compile_model()

# train the model
modelTrainer.train_model(runName, path_to_train_mix, path_to_train_speech, \
                         path_to_val_mix, path_to_val_speech)



