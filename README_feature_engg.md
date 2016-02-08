# feature_engg help

This document explains the working of the 'feature_engg' module. The aim of this module is to provide you with useful functions for transforming data based on pre-defined plots. Also, basic level imputation is supported.

This module has just 3 function:

####`bivariate(col1, col2, col1Type='infer', col2Type='infer')`
  - used to generate plots corresponding to a pair of columns
  - Arguments: 
    1. col1: Series object - the first column to be analyzed 
    2. col2: Series object - the second column to be analyzed
    3. col1Type: the type - 'categorical' or 'continuous' of the column 1. If nothing specified, it will be automatically infered. This is provided to account for categorical variables like 'H-M-L' which are coded as '0-1-2' and will be treated numerical by default.
    4. col2Type: same as col1Type but for column 2.

  - Output: 
    1. Chart corresponding to the data types:
      - Both categorical variables: confusion matrix and bar charts with absolute and percentage values
      - Both continuous variables: scatter chart with regression line
      - One categorical and one continuous: box-and-whiskers plot 

####`univariate(col, colType='infer', transformation=None, param={},check_duplicate_categories=False,return_mod=False)`
  - used to analuze 1 variable at a time. Note: This doesn't change the original variable 
  - Arguments: 
    1. col: Series object - the column to be analyzed 
    2. colType: string constant - 2 options: 'categorical' or 'continuous' 
    3. transformation: string constant - depending on colType, it can be None or one of following:
      - 'continuous' : log, square, square root, cube, cube root, combine
      - 'categorical' : combine
    4. param: the parameters required for the transformation type selected
      - if 'continuous' & 'combine': Pass a list of intermediate cut-points. Min and Max will automatically added
      - if 'categorical' & 'combine' : Pass a dictionary in format - {'new_category':[list of categories to combines]}
    5. check_duplicate_categories: Applicable only for categorical varaible. Checks if the categories are different only by upper or lower case and resolves the same. Eg - 'High' and 'high' will be resolved to 'high'
    6. return_mod: if True returns the modified variable which can be used to create a new variable or replace the old one
    
  - Output: 
    1. Chart corresponding to the data
    2. The modified variable if return_mod is True

####`imputation(data, col, method, param={}, colType='infer')`
  - Function to perform imputation. 
  - Arguments:
    1. data: the full data frame whose column is to be modified
    2. col: name of the column to be modified
    3. method: differs depending on type of variable
      - for continuous:
        1. mean - impute by mean
        2. median - impute by median
      -for categorical:
        1. mode - impute by mode
        2. category - fixed value impute
    4. param: dictionary of the additional parameters to be used:
      - 'groupby':colname - the colum by which the method is to be grouped for imputation
    5. colType: type of the column. if it is 'infer', it will be selected based on datatype
