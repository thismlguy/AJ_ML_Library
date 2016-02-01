## AJ_ML_Library

This library contains some functions designed to make exploration, feature engineering and model building easy. The core runs on Pandas and Sklearn but the functions perform multiple tasks at the same time.

The various files in the module are:

1. __init__.py
  - This is required so that python recognizes this folder as a module
  - It is a blank file 
   
2. setup.py
  - This file contains the pre-requisite variable definitions
  - It should be updates just once at the time of set-up
  
3. exploration.py
  - This module is used for quick first time view of data
  - It generates an excel file with various summary statistics
  - Check out README_exploration for details
  
4. models_classification.py
  - This module contains an sklearn wrapper for different algorithms
  - It can used for performing multiple tasks with a single line of code
  - Check out README_models_classification for details
  

## Setting Up:

There are 2 ways to use this module:

### 1. Copy files in current working directory
I don't recommend this but its an easy trial option. Just download the module which is required to be used in the current working directory and call it.

### 2. Setup as a permanent module

This is a more efficient method and allows invocation anywhere like standard python packages. There can be multiple ways to do this but I have done it in an easy way:

1. Create a new folder in the 'site-packages' module in Python. 
  - I use Max OS X and for me the path is: '/Users/aarshay/anaconda/lib/python2.7/site-packages/AJ_ML_Library'
  - Navigate to yours and create new folder
  - Remember that this is the path to be entered in the 'setup.py' file
2. Import library as:
  - from AJ_ML_Library import models_classification

Note that the same can be done by cloning the repository in any local folder and adding that to the Python path. Choose the method which works best for you.

## Upcoming Updates:

1. feature_engg module - for performing feature engineering
2. models_regression module - similar to classification but for regression
3. improved xgboost support - added functionality
