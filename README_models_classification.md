## models_classification help

This module is a Sklearn wrapper for running various classification algorithms using the pre-defined functions. The algirthms supported are:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Adaboost
5. Gradient Boosting Machine
6. Xgboost

Along with algorithms, this module provides an easy way to create ensemble models by providing options of storing the models in a particular format and then easily selecting which ones to combine.

Each algorithm is defined as a class and same function names are defined for each class. The various steps involved in using the classes are:

### Initialization

The different ways of initializing algorithms are:

1. Logistic_Regression(data_train, data_test, target, predictors, cvfolds=5, scoring_metric='accuracy')
2. Decision_Tree_Class(data_train, data_test, target, predictors, cvfolds=5, scoring_metric='accuracy')
3. Random_Forest_Class(data_train, data_test, target, predictors, cvfolds=5, scoring_metric='accuracy')
4. AdaBoost_Class(data_train, data_test, target, predictors, cvfolds=5, scoring_metric='accuracy')
5. GradientBoosting_Class(data_train, data_test, target, predictors, cvfolds=5, scoring_metric='accuracy')
6. XGBoost_Class(data_train, data_test, target, predictors, cvfolds=5, scoring_metric_skl='accuracy', scoring_metric_xgb='error')

Note: 

1. The parameters are not defined at this stage, but the data is defined. Now we can perform multiple operations on the same data without having to specify it again and again.
2. XGBoost has different parameters because it is not part of sklearn library and has different parameters and options

Now let's look at the different functions for each class. Most of the functions are generic accross models. I'll mention the class specific functions in the end if any. Remember to call the functions using classname.function(..)

### Setting Parameters:

Next step is to define parameters. Since it is based on sklearn, I've kept the model parameters exactly same as that of sklearn. So the definition of parameters of each model can be found on sklean help.

The function definition is:

`set_parameters(param=None, set_default=False)`

The arguments are:

1. param: 
  - This is a dictionary with keys as the parameter names and values as the required value of parameter. 
  - Note that ONLY the parameters which are requried to be updated should be passed. All others will be kept as default or last modified values. 
  - Along with the model parameters, another parameter 'cv_folds' can be passed to set the number of cross-validation folds to be used in the  model.
  
2. set_default: 
  - Pass a 'True' value to overwrite all parameters to default value. 
  - If this is True, then the param value is ignored even if it is passed

### Fitting the Model

The model can be fit by calling the following function:

`modelfit(performCV=True, printTopN='all')`

- This function will fit the model and also perform predictions on the test set. 
- The function will print the following model characteristics as well:
  1. Confusion Matrix
  2. Accuracy
  3. Specified Scoring Metric
  4. Mean and Standard deviation of Test accuracy of CV folds (only is performCV is True)

parameters:
1. performCV: If not set to False, it will perform K-fold cross-validation using the same number of folds as passed in the 'cv_folds' parameter. Default value is 5 if not set by user.
2. printTopN: If a number set, it'll display only top those many variables in feature importance plot

The XGBoost class has a different implementation:
`modelfit(performCV=True, useTrainCV=False, TrainCVFolds=5, early_stopping_rounds=20, show_progress=True, printTopN='all')`

The different parameters are:
1. performCV: similar to above
2. useTrainCV: whether to use "cv" function of xgboost package to change the 'n_estimators' to the value selected from cv run
3. early_stopping_rounds: input for the cv function. applicable only if useTrainCV is True
4. show_progress: input for the cv function. applicable only if useTrainCV is True
  
### Recursive Feature Elimination (RFE)

There are two functions to perform RFE:

1. `RecursiveFeatureElimination(nfeat=None, step=1, inplace=False)`

This function will perform Recursive Feature Elimination on the dataset. Look in sklearn help for details on RFE. The argumetns are:

1. nfeat: the number of features to select
2. step: the number of features to remove at each step
3. inplace: if True, function will replace the predictors in the model with the ones selected by this algorithm

2. `RecursiveFeatureEliminationCV(step=1, inplace=False, scoring='accuracy')`

This function perform cross-validation at each step and then reduces the least significant variable. Then the features set with highest cross-validation score is selected. The arguments are:

1. step: the number of features to remove at each step
2. inplace: if True, function will replace the predictors in the model with the ones selected by this algorithm
3. scoring: the scoring metric to choose for CV. Look at sklean metrics for other options

### Export predictions (Submission)

If you are participating in a ML competition, this function will help you create submissions easily. The function is:

`submission(self, IDcol, filename="Submission.csv")`

This will export a csv file with 2 columns - ID and target variable. The arguments are:

1. IDcol: The name of the column in dataframe to be used as the ID column. Note that the target feature is already defined in model. 
2. filename: The name of the the exported file. Default is 'Submission.csv'

`submission_proba(IDcol, proba_colnames, filename="Submission.csv")` 
This will perform similar function as above but will ouput the predicted probability. It has one additional option:
3. proba_colnames: A list of colnames in the order of classes which will be used for defining the names of the columns of outcomes of individual probabilities


### Support Functions:

Some functions are defined to help in some intermediate analysis. These are:

`set_predictors(predictors)`
This can be used to update the predictors in the model

`get_predictors()`
This will return a list of predictors being used in the model

`get_test_predictions(getprob=False)`
Returns a Series object with the predictions on test data. If 'getprob' is true, then the probabilities will be returned.

`get_feature_importance()`
Returns a Series object with feture importances. Only works for models which give feature importance as output.

### Exporting Models for Ensemble

This function will create a log of the models of each type which will help in creating an ensemble later on. The function deifnition is:

`export_model(IDcol)`

Here IDcol is the name of the column to be used as the ID variable. This function does following:

1. Creates an 'ensemble' directory (folder) in your current working directory if one doesn't already exists
2. Assigns a unique model ID to the exported model.
3. Uses the ID and updates (or creates if doesn't exist) the model specific csv file which contains the parameters used for creating a model and the results (accuracy, cv score, etc)
4. Creates a submission file, inside ensemble directory, for the model with the Unique ID created.

The details of how this exported model will be used are discussed next.

## Creating an Ensemble:

A separate class if defined for creating ensemble models. This will use the models exported using 'export_model' function of the classificiation algorithm classes. The various steps involved are:

1. Look at the the model log files in 'ensemble' directory. These are named very intuitively, eg: 'logreg_models.csv', 'rf_models.csv','gbm_models.csv', etc. Select the desired model IDs from each algo to consider for ensembling.
2. Initialize the ensemble class using - Ensemble_Classification()
3. Set the models to be considered for ensemble using function create_ensemble_data
4. Check the correlation between chosen models using check_ch2 or check_diff and select ones to include finally
5. Create final emsemble models using selected models.

The details of functions used at each step are:

### Initialize class:

`Ensemble_Classification(target, IDcol)`

This will help in creating an instance of class. target and IDcol are the features to be used as the target and ID.

### Selecting Initial set of Models

Having decided the model IDs of each type to be considered, use the following function:
`create_ensemble_data(models)`

The argument 'models' is a dictionary with keys as the keyword for each model and value as the list of model IDs for each kind. Example:
models = {'logreg':[1,2,3], 'dectree':[3,4], 'rf':[1,2,3,4,5], 'gbm':[1,2,3], 'xgboost':[1,2]}

This will select following:
1. model IDs 1,2,3 for logistic regression,
2. model IDs 3,4 for decision tree,
3. model IDs 1,2,3,4,5 for random forest, and so on..

### Analyzing correlation:

The correlation between each model can be checked using 2 ways:

`check_ch2()`
This gives a matrix with the chi-square p-value between each pair

`check_diff()`
This gives a matrix with the fraction of different classifications between each pair. 0.2 would mean 20% values are classified differently by 2 models.

### Final ensemble

The final ensemble can be created using function:
`submission(models_to_use=None, filename="Submission_ensemble.csv")`

Arguments:
1. models_to_use: a list with full model names to use, eg: ['logreg_1','logreg_3','rf_3','gbm_4','gbm_5',...]. If None, all models are used
2. filename: name of the final submission file. This will have 2 columns - IDcol and final predictions after ensemble.

Note: ensemble takes the mode of the predictions of all algorithms  






