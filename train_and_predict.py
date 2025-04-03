#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train all models (traditional machine learning models and deep learning models)
"""
import os
import sys

# Add current directory to path
sys.path.append(os.path.abspath('.'))

if __name__ == "__main__":
    # Train traditional models
    os.system('python -m src.main train --data_dir realdata --output_dir models')
    
    # Train deep learning models
    os.system('python train_deep_model.py') 