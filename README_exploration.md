# exploration help

This document explains the working of the 'exploration' module. The aim of this module is to provide you with useful information about any dataset in a MS Excel format.

Note: remember to set the 'module_location' in the setup.py file before running this module.

This module has just 1 function:

`exploration(data_train, data_test, outcome)`

Arguments:
1. data_train: the training data (Pandas DataFrame)
2. data_test: the testing data {Pandas DataFrame)
3. outcome: name of the outcome variable

Additional Requirement:
1. datatypes.csv file in the current working directory. It should have 2 columns: 
  - feature: the name of the different variables/features in the model
  - type: thir data type with 2 possible values - 'continuous' or 'categorical'
Note: The code won't work without this file.

Outcome:
A file named 'DataAnalysis.xlsx' in the current working directory. This file will have 3 sheets:

1. Overall Summary:
  - Gives basic information like number of categorical and continuous features
  - Also gives number of unique values

2. Categorical Summary:
  - Gives summary of categorical features
  - The summary statistics for each feature include:
    - #Unique Values
    - #Missing Values
    - Significance : p-value with outcome var. The statistical test used depends on datatype of each variable.
    - 20% concentration: #labels comprising top 20% data
    - 40% concentration: #labels comprising top 40% data
    - 60% concentration: #labels comprising top 60% data
    - 80% concentration: #labels comprising top 80% data
    - Top 5 Categories: names along with #data points in each

3. Continuous Summary:
  - Gives summary of continuous features
  - The summary statistics for each feature include:
    - #Missing Values
    - Significance : p-value with outcome var. The statistical test used depends on datatype of each variable.
    - Mean
    - Standard Deviation
    - Min
    - 25th percentile
    - Media
    - 75th Percentile
    - Max
    - #Values beyond 1.5 IQR

4. Relational Summary:
  - A matrix with p-values for each combination of variables.
  - The statistical test performed are:
    - 2 Categorical: chi-square
    - 2 Continuous: Pearson's correlation
    - 1 Categorical, 1 Continuous: ANOVA
    
This excel file can give you a very good sense of different features in a dataset and will save a lot of time. This is mostly useful while looking at a new data set for the first time.
