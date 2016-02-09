#######################################################################################################################
##### IMPORT STANDARD MODULES
#######################################################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydot
import os
from scipy.stats import mode
        
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn import metrics
from sklearn.feature_selection import RFE, RFECV
from sklearn.externals.six import StringIO  

#######################################################################################################################
##### GENERIC MODEL CLASS
#######################################################################################################################

#This class contains the generic regression functions and variable definitions applicable across all models 
class GenericModelRegr(object):
    def __init__(self, alg, data_train, data_test, target, predictors=[], output_transformation=None):
        self.alg = alg                  #an instance of particular model class
        self.data_train = data_train    #training data
        self.data_test = data_test      #testing data
        self.target = target
        self.cv_folds = 10  #Default 10. can be set using set_parameters
        self.predictors = predictors
        self.train_predictions = []
        self.test_predictions_raw = []
        self.test_predictions_transformed = []
        if output_transformation is None:
            self.output_transformation = lambda x: x
        else:
            self.output_transformation = output_transformation

        #Define a Series object to store generic regression model outcomes; 
        self.regression_output=pd.Series(index=['ModelID','RMSE','CVScore_mean','CVScore_std',
                                             'ActualScore (manual entry)','CVMethod','Predictors'])
    
    #Modify and get predictors for the model:
    def set_predictors(self, predictors):
        self.predictors=predictors

    def get_predictors(self):
        return self.predictors

    def get_test_predictions(self):
        return self.test_predictions_transformed

    #Implement K-Fold cross-validation 
    def KFold_CrossValidation(self):
        # Generate cross validation folds for the training dataset.  
        kf = KFold(self.data_train.shape[0], n_folds=self.cv_folds)
        
        error = []
        for train, test in kf:
            # Filter training data
            train_predictors = (self.data_train[self.predictors].iloc[train,:])
            # The target we're using to train the algorithm.
            train_target = self.data_train[self.target].iloc[train]
            # Training the algorithm using the predictors and target.
            self.alg.fit(train_predictors, train_target)
        
            #Record error from each cross-validation run
            error.append(np.sqrt(metrics.mean_squared_error( self.output_transformation(self.data_train[self.target].iloc[test].values) ,
                                 self.output_transformation(self.alg.predict(self.data_train[self.predictors].iloc[test,:])) )))
        
        return {'mean_error': np.mean(error),
                'std_error': np.std(error),
                'min_error': np.min(error),
                'max_error':np.max(error),
                'all_error': error }

    #Implement recursive feature elimination
    # Inputs:
    #     nfeat - the num of top features to select
    #     step - the number of features to remove at each step
    #     inplace - True: modiy the data of the class with the data after RFE
    # Returns:
    #     selected - a series object containing the selected features
    def RecursiveFeatureElimination(self, nfeat=None, step=1, inplace=False):
        
        rfe = RFE(self.alg, n_features_to_select=nfeat, step=step)
        
        rfe.fit(self.data_train[self.predictors], self.data_train[self.target])
        
        ranks = pd.Series(rfe.ranking_, index=self.predictors)
        
        selected = ranks.loc[rfe.support_]

        if inplace:
            self.set_predictors(selected.index.tolist())
        
        return selected

    #Performs similar function as RFE but with CV. It removed features similar to RFE but the importance of the group of features is based on the cross-validation score. The set of features with highest cross validation scores is then chosen. The difference from RFE is that the #features is not an input but selected by algo
    def RecursiveFeatureEliminationCV(self, step=1, inplace=False, scoring='mean_squared_error'):
        rfecv = RFECV(self.alg, step=step,cv=self.cv_folds,scoring=scoring)
        
        rfecv.fit(self.data_train[self.predictors], self.data_train[self.target])

        min_nfeat = len(self.predictors) - step*(len(rfecv.grid_scores_)-1)  # n - step*(number of iter - 1)
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score")
        plt.plot(range(min_nfeat, len(self.predictors) + 1,step), rfecv.grid_scores_)
        plt.show(block=False)

        ranks = pd.Series(rfecv.ranking_, index=self.predictors)
        selected = ranks.loc[rfecv.support_]

        if inplace:
            self.set_predictors(selected.index.tolist())
        return ranks

    # Determine key metrics to analyze the regression model. These are stored in the regression_output series object belonginf to this class.
    def calc_model_characteristics(self):
        self.regression_output['RMSE'] = np.sqrt(metrics.mean_squared_error(self.output_transformation(self.data_train[self.target].values),
                                                                            self.output_transformation(self.train_predictions)))
        cv_score= self.KFold_CrossValidation()
        self.regression_output['CVMethod'] = 'KFold - ' + str(self.cv_folds)
        self.regression_output['CVScore_mean'] = cv_score['mean_error']
        self.regression_output['CVScore_std'] = cv_score['std_error']
        self.regression_output['CVScore_max'] = cv_score['max_error']
        self.regression_output['CVScore_min'] = cv_score['min_error']
        self.regression_output['Predictors'] = str(self.predictors)

    # Print the metric determined in the previous function.
    def printReport(self):
        print "\nModel Report"
        print "RMSE : %.4g" % self.regression_output['RMSE']
        print "CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (self.regression_output['CVScore_mean'],self.regression_output['CVScore_std'],
                                                                                self.regression_output['CVScore_min'], self.regression_output['CVScore_max'])
    
    #Define the function which will be used to transform the values of the output variable into submission output
    def set_output_transformation(self,f):
        self.output_transformation = f

    # create submission file
    def submission(self, IDcol,filename="Submission.csv", submission_target_name=None):
        if submission_target_name is None:
            submission_target_name=self.target

        self.data_test[submission_target_name] = self.test_predictions_transformed
        IDcol = list(IDcol)
        IDcol.append(submission_target_name)
        submission = pd.DataFrame({ x: self.data_test[x] for x in IDcol})
        submission.to_csv(filename, index=False)

    #checks whether the ensemble directory exists and creates one if it doesn't
    def create_ensemble_dir(self):
        ensdir = os.path.join(os.getcwd(), 'ensemble')
        if not os.path.isdir(ensdir):
            os.mkdir(ensdir)

#######################################################################################################################
##### LINEAR REGRESSION
#######################################################################################################################

class Linear_Regression(GenericModelRegr):

    def __init__(self, data_train, data_test, target, predictors=[], output_transformation=None):
        GenericModelRegr.__init__(self, alg=LinearRegression(), data_train=data_train, data_test=data_test, target=target, 
                                    predictors=predictors, output_transformation=output_transformation)
        self.default_parameters = {'fit_intercept':True, 'normalize':False, 'copy_X':True,'n_jobs':1}
        self.model_output=pd.Series(self.default_parameters)
        self.model_output['Coefficients'] = "-"
        
        #Set parameters to default values:
        self.set_parameters(set_default=True)

    # Set the parameters of the model. 
    # Note: 
    #     > only the parameters to be updated are required to be passed
    #     > if set_default is True, the passed parameters are ignored and default parameters are set which are defined in   scikit learn module
    def set_parameters(self, param=None, set_default=False):        
        if set_default:
            param = self.default_parameters

        if 'fit_interept' in param:
            self.alg.set_params(fit_intercept=param['fit_intercept'])
            self.model_output['fit_intercept'] = param['fit_intercept']

        if 'normalize' in param:
            self.alg.set_params(normalize=param['normalize'])
            self.model_output['normalize'] = param['normalize']

        if 'copy_X' in param:
            self.alg.set_params(copy_X=param['copy_X'])
            self.model_output['copy_X'] = param['copy_X']

        if 'n_jobs' in param:
            self.alg.set_params(n_jobs=param['n_jobs'])
            self.model_output['n_jobs'] = param['n_jobs']

        if 'cv_folds' in param:
            self.cv_folds = param['cv_folds']
    
    #Fit the model using predictors and parameters specified before.
    def modelfit(self):

        #Outpute the parameters for the model for cross-checking:
        print 'Model being built with the following parameters:'
        print self.alg.get_params()

        self.alg.fit(self.data_train[self.predictors], self.data_train[self.target])
        # print self.alg.intercept_,self.alg.coef_
        coeff = pd.Series(np.concatenate(([self.alg.intercept_],self.alg.coef_)), index=["Intercept"]+self.predictors)
        print 'Coefficients: '
        print coeff.sort_values()

        self.model_output['Coefficients'] = coeff.to_string()
        
        #Get predictions:
        self.train_predictions = self.alg.predict(self.data_train[self.predictors])
        self.test_predictions_raw = self.alg.predict(self.data_test[self.predictors])
        self.test_predictions_transformed = self.output_transformation(self.test_predictions_raw)

        self.calc_model_characteristics()
        self.printReport()
    
    #Export the model into the model file as well as create a submission with model index. This will be used for creating an ensemble.
    def export_model(self, IDcol, submission_target_name=None):
        self.create_ensemble_dir()
        filename = os.path.join(os.getcwd(),'ensemble/linreg_models.csv')
        comb_series = self.regression_output.append(self.model_output, verify_integrity=True)

        if os.path.exists(filename):
            models = pd.read_csv(filename)
            mID = int(max(models['ModelID'])+1)
        else:
            mID = 1
            models = pd.DataFrame(columns=comb_series.index)
            
        comb_series['ModelID'] = mID
        models = models.append(comb_series, ignore_index=True)
        
        models.to_csv(filename, index=False, float_format="%.5f")
        model_filename = os.path.join(os.getcwd(),'ensemble/linreg_'+str(mID)+'.csv')
        self.submission(IDcol, model_filename, submission_target_name)

#######################################################################################################################
##### RIDGE REGRESSION
#######################################################################################################################

class Ridge_Regression(GenericModelRegr):

    def __init__(self, data_train, data_test, target, predictors=[], output_transformation=None):
        GenericModelRegr.__init__(self, alg=Ridge(), data_train=data_train, data_test=data_test, target=target, 
                                    predictors=predictors, output_transformation=output_transformation)
        self.default_parameters = {'alpha':1.0,'fit_intercept':True, 'normalize':False, 'copy_X':True,'max_iter':None,
                                    'tol':0.001, 'solver':'auto','random_state':None}
        self.model_output=pd.Series(self.default_parameters)
        self.model_output['Coefficients'] = "-"
        
        #Set parameters to default values:
        self.set_parameters(set_default=True)

    # Set the parameters of the model. 
    # Note: 
    #     > only the parameters to be updated are required to be passed
    #     > if set_default is True, the passed parameters are ignored and default parameters are set which are defined in   scikit learn module
    def set_parameters(self, param=None, set_default=False):        
        if set_default:
            param = self.default_parameters

        if 'alpha' in param:
            self.alg.set_params(alpha=param['alpha'])
            self.model_output['alpha'] = param['alpha']

        if 'fit_interept' in param:
            self.alg.set_params(fit_intercept=param['fit_intercept'])
            self.model_output['fit_intercept'] = param['fit_intercept']

        if 'normalize' in param:
            self.alg.set_params(normalize=param['normalize'])
            self.model_output['normalize'] = param['normalize']

        if 'copy_X' in param:
            self.alg.set_params(copy_X=param['copy_X'])
            self.model_output['copy_X'] = param['copy_X']

        if 'max_iter' in param:
            self.alg.set_params(max_iter=param['max_iter'])
            self.model_output['max_iter'] = param['max_iter']

        if 'tol' in param:
            self.alg.set_params(tol=param['tol'])
            self.model_output['tol'] = param['tol']

        if 'solver' in param:
            self.alg.set_params(solver=param['solver'])
            self.model_output['solver'] = param['solver']

        if 'random_state' in param:
            self.alg.set_params(random_state=param['random_state'])
            self.model_output['random_state'] = param['random_state']

        if 'cv_folds' in param:
            self.cv_folds = param['cv_folds']
    
    #Fit the model using predictors and parameters specified before.
    def modelfit(self):

        #Outpute the parameters for the model for cross-checking:
        print 'Model being built with the following parameters:'
        print self.alg.get_params()

        self.alg.fit(self.data_train[self.predictors], self.data_train[self.target])
        # print self.alg.intercept_,self.alg.coef_
        coeff = pd.Series(np.concatenate(([self.alg.intercept_],self.alg.coef_)), index=["Intercept"]+self.predictors)
        print 'Coefficients: '
        print coeff.sort_values()

        self.model_output['Coefficients'] = coeff.to_string()
        
        #Get predictions:
        self.train_predictions = self.alg.predict(self.data_train[self.predictors])
        self.test_predictions_raw = self.alg.predict(self.data_test[self.predictors])
        self.test_predictions_transformed = self.output_transformation(self.test_predictions_raw)

        self.calc_model_characteristics()
        self.printReport()
    
    #Export the model into the model file as well as create a submission with model index. This will be used for creating an ensemble.
    def export_model(self, IDcol, submission_target_name=None):
        self.create_ensemble_dir()
        filename = os.path.join(os.getcwd(),'ensemble/ridge_models.csv')
        comb_series = self.regression_output.append(self.model_output, verify_integrity=True)

        if os.path.exists(filename):
            models = pd.read_csv(filename)
            mID = int(max(models['ModelID'])+1)
        else:
            mID = 1
            models = pd.DataFrame(columns=comb_series.index)
            
        comb_series['ModelID'] = mID
        models = models.append(comb_series, ignore_index=True)
        
        models.to_csv(filename, index=False, float_format="%.5f")
        model_filename = os.path.join(os.getcwd(),'ensemble/ridge_'+str(mID)+'.csv')
        self.submission(IDcol, model_filename, submission_target_name)

#######################################################################################################################
##### LASSO REGRESSION
#######################################################################################################################

class Lasso_Regression(GenericModelRegr):

    def __init__(self, data_train, data_test, target, predictors=[], output_transformation=None):
        GenericModelRegr.__init__(self, alg=Lasso(), data_train=data_train, data_test=data_test, target=target, 
                                    predictors=predictors, output_transformation=output_transformation)
        self.default_parameters = {'alpha':1.0,'fit_intercept':True, 'normalize':False, 'copy_X':True,'max_iter':None,
                                    'tol':0.0001, 'solver':'auto','random_state':None, 'precompute':False, 'warm_start':False,
                                    'positive':False, 'selection':'cyclic'}
        self.model_output=pd.Series(self.default_parameters)
        self.model_output['Coefficients'] = "-"
        
        #Set parameters to default values:
        self.set_parameters(set_default=True)

    # Set the parameters of the model. 
    # Note: 
    #     > only the parameters to be updated are required to be passed
    #     > if set_default is True, the passed parameters are ignored and default parameters are set which are defined in   scikit learn module
    def set_parameters(self, param=None, set_default=False):        
        if set_default:
            param = self.default_parameters

        if 'alpha' in param:
            self.alg.set_params(alpha=param['alpha'])
            self.model_output['alpha'] = param['alpha']

        if 'fit_interept' in param:
            self.alg.set_params(fit_intercept=param['fit_intercept'])
            self.model_output['fit_intercept'] = param['fit_intercept']

        if 'normalize' in param:
            self.alg.set_params(normalize=param['normalize'])
            self.model_output['normalize'] = param['normalize']

        if 'copy_X' in param:
            self.alg.set_params(copy_X=param['copy_X'])
            self.model_output['copy_X'] = param['copy_X']

        if 'max_iter' in param:
            self.alg.set_params(max_iter=param['max_iter'])
            self.model_output['max_iter'] = param['max_iter']

        if 'tol' in param:
            self.alg.set_params(tol=param['tol'])
            self.model_output['tol'] = param['tol']

        if 'solver' in param:
            self.alg.set_params(solver=param['solver'])
            self.model_output['solver'] = param['solver']

        if 'random_state' in param:
            self.alg.set_params(random_state=param['random_state'])
            self.model_output['random_state'] = param['random_state']

        if 'precompute' in param:
            self.alg.set_params(precompute=param['precompute'])
            self.model_output['precompute'] = param['precompute']

        if 'warm_start' in param:
            self.alg.set_params(warm_start=param['warm_start'])
            self.model_output['warm_start'] = param['warm_start']

        if 'positive' in param:
            self.alg.set_params(positive=param['positive'])
            self.model_output['positive'] = param['positive']

        if 'selection' in param:
            self.alg.set_params(selection=param['selection'])
            self.model_output['selection'] = param['selection']

        if 'cv_folds' in param:
            self.cv_folds = param['cv_folds']
    
    #Fit the model using predictors and parameters specified before.
    def modelfit(self):

        #Outpute the parameters for the model for cross-checking:
        print 'Model being built with the following parameters:'
        print self.alg.get_params()

        self.alg.fit(self.data_train[self.predictors], self.data_train[self.target])
        # print self.alg.intercept_,self.alg.coef_
        coeff = pd.Series(np.concatenate(([self.alg.intercept_],self.alg.coef_)), index=["Intercept"]+self.predictors)
        print 'Coefficients: '
        print coeff.sort_values()

        self.model_output['Coefficients'] = coeff.to_string()
        
        #Get predictions:
        self.train_predictions = self.alg.predict(self.data_train[self.predictors])
        self.test_predictions_raw = self.alg.predict(self.data_test[self.predictors])
        self.test_predictions_transformed = self.output_transformation(self.test_predictions_raw)

        self.calc_model_characteristics()
        self.printReport()
    
    #Export the model into the model file as well as create a submission with model index. This will be used for creating an ensemble.
    def export_model(self, IDcol, submission_target_name=None):
        self.create_ensemble_dir()
        filename = os.path.join(os.getcwd(),'ensemble/lasso_models.csv')
        comb_series = self.regression_output.append(self.model_output, verify_integrity=True)

        if os.path.exists(filename):
            models = pd.read_csv(filename)
            mID = int(max(models['ModelID'])+1)
        else:
            mID = 1
            models = pd.DataFrame(columns=comb_series.index)
            
        comb_series['ModelID'] = mID
        models = models.append(comb_series, ignore_index=True)
        
        models.to_csv(filename, index=False, float_format="%.5f")
        model_filename = os.path.join(os.getcwd(),'ensemble/lasso_'+str(mID)+'.csv')
        self.submission(IDcol, model_filename, submission_target_name)


#######################################################################################################################
##### DECISION TREE
#######################################################################################################################

class Decision_Tree_Regr(GenericModelRegr):
    def __init__(self, data_train, data_test, target, predictors=[], output_transformation=None):
        GenericModelRegr.__init__(self, alg=DecisionTreeRegressor(), data_train=data_train, data_test=data_test, target=target, 
                                    predictors=predictors, output_transformation=output_transformation)
        self.default_parameters = {'criterion':'mse', 'max_depth':None, 
                                   'min_samples_split':2, 'min_samples_leaf':1, 
                                   'max_features':None, 'random_state':None, 'max_leaf_nodes':None}
        self.model_output = pd.Series(self.default_parameters)
        self.model_output['Feature_Importance'] = "-"

        #Set parameters to default values:
        self.set_parameters(set_default=True)


    # Set the parameters of the model. 
    # Note: 
    #     > only the parameters to be updated are required to be passed
    #     > if set_default is True, the passed parameters are ignored and default parameters are set which are defined in   scikit learn module
    def set_parameters(self, param=None, set_default=False):
        
        #Set param to default values if default set
        if set_default:
            param = self.default_parameters

        if 'criterion' in param:
            self.alg.set_params(criterion=param['criterion'])
            self.model_output['criterion'] = param['criterion']

        if 'max_depth' in param:
            self.alg.set_params(max_depth=param['max_depth'])
            self.model_output['max_depth'] = param['max_depth']

        if 'min_samples_split' in param:
            self.alg.set_params(min_samples_split=param['min_samples_split'])
            self.model_output['min_samples_split'] = param['min_samples_split']

        if 'min_samples_leaf' in param:
            self.alg.set_params(min_samples_leaf=param['min_samples_leaf'])
            self.model_output['min_samples_leaf'] = param['min_samples_leaf']

        if 'max_features' in param:
            self.alg.set_params(max_features=param['max_features'])
            self.model_output['max_features'] = param['max_features']

        if 'random_state' in param:
            self.alg.set_params(random_state=param['random_state'])
            self.model_output['random_state'] = param['random_state']

        if 'max_leaf_nodes' in param:
            self.alg.set_params(max_leaf_nodes=param['max_leaf_nodes'])
            self.model_output['max_leaf_nodes'] = param['max_leaf_nodes']

        if 'cv_folds' in param:
            self.cv_folds = param['cv_folds']
    
    #Fit the model using predictors and parameters specified before.
    def modelfit(self, printROC=False):

        #Outpute the parameters for the model for cross-checking:
        print 'Model being built with the following parameters:'
        print self.alg.get_params()

        self.alg.fit(self.data_train[self.predictors], self.data_train[self.target])
        
        print 'Feature Importance Scores: '
        featimp = pd.Series(self.alg.feature_importances_, index=self.predictors).sort_values()
        print featimp
        self.model_output['Feature_Importance'] = featimp.to_string()

        #Get predictions:
        self.train_predictions = self.alg.predict(self.data_train[self.predictors])
        self.test_predictions_raw = self.alg.predict(self.data_test[self.predictors])
        self.test_predictions_transformed = self.output_transformation(self.test_predictions_raw)

        self.calc_model_characteristics()
        self.printReport()

    #Print the tree in visual format
    # Inputs:
    #     export_pdf - if True, a pdf will be exported with the filename as specified in pdf_name argument
    #     pdf_name - name of the pdf file if export_pdf is True
    def printTree(self, export_pdf=True, pdf_name="Decision_Tree.pdf"):
        dot_data = StringIO() 
        export_graphviz(self.alg, out_file=dot_data, feature_names=self.predictors,    
                         filled=True, rounded=True, special_characters=True) 
        graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
        
        if export_pdf:
            graph.write_pdf(pdf_name)

        return graph

    #Export the model into the model file as well as create a submission with model index. This will be used for creating an ensemble.
    def export_model(self, IDcol, submission_target_name=None):
        self.create_ensemble_dir()
        filename = os.path.join(os.getcwd(),'ensemble/dectree_models.csv')
        comb_series = self.regression_output.append(self.model_output, verify_integrity=True)

        if os.path.exists(filename):
            models = pd.read_csv(filename)
            mID = int(max(models['ModelID'])+1)
        else:
            mID = 1
            models = pd.DataFrame(columns=comb_series.index)
            
        comb_series['ModelID'] = mID
        models = models.append(comb_series, ignore_index=True)
        
        models.to_csv(filename, index=False, float_format="%.5f")
        model_filename = os.path.join(os.getcwd(),'ensemble/dectree_'+str(mID)+'.csv')
        self.submission(IDcol, model_filename, submission_target_name)

#######################################################################################################################
##### RANDOM FOREST
#######################################################################################################################

class Random_Forest_Regr(GenericModelRegr):
    def __init__(self, data_train, data_test, target, predictors=[], output_transformation=None):
        GenericModelRegr.__init__(self, alg=RandomForestRegressor(), data_train=data_train, data_test=data_test, target=target, 
                                    predictors=predictors, output_transformation=output_transformation)
        self.default_parameters = {
                                    'n_estimators':10, 'criterion':'mse', 'max_depth':None,' min_samples_split':2, 
                                    'min_samples_leaf':1, 'max_features':'auto', 'max_leaf_nodes':None,
                                    'oob_score':False, 'random_state':None 
                                  }
        self.model_output = pd.Series(self.default_parameters)
        self.model_output['Feature_Importance'] = "-"
        self.model_output['OOB_Score'] = "-"

        #Set parameters to default values:
        self.set_parameters(set_default=True)


    # Set the parameters of the model. 
    # Note: 
    #     > only the parameters to be updated are required to be passed
    #     > if set_default is True, the passed parameters are ignored and default parameters are set which are defined in   scikit learn module
    def set_parameters(self, param=None, set_default=False):
        
        #Set param to default values if default set
        if set_default:
            param = self.default_parameters

        #trees in forest:
        if 'n_estimators' in param:
            self.alg.set_params(n_estimators=param['n_estimators'])
            self.model_output['n_estimators'] = param['n_estimators']
        
        #decision tree split criteria
        if 'criterion' in param:
            self.alg.set_params(criterion=param['criterion'])
            self.model_output['criterion'] = param['criterion']
        
        #maximum depth of each tree (ignored if max_leaf_nodes is not None)
        if 'max_depth' in param:
            self.alg.set_params(max_depth=param['max_depth'])
            self.model_output['max_depth'] = param['max_depth']
        
        #min #samples required to split an internal node; typically around 20-50
        if 'min_samples_split' in param:
            self.alg.set_params(min_samples_split=param['min_samples_split'])
            self.model_output['min_samples_split'] = param['min_samples_split']
        
        #The minimum number of samples in newly created leaves.
        if 'min_samples_leaf' in param:
            self.alg.set_params(min_samples_leaf=param['min_samples_leaf'])
            self.model_output['min_samples_leaf'] = param['min_samples_leaf']

        #max features to be considered for each split
        if 'max_features' in param:
            self.alg.set_params(max_features=param['max_features'])
            self.model_output['max_features'] = param['max_features']

        #for replication of results
        if 'random_state' in param:
            self.alg.set_params(random_state=param['random_state'])
            self.model_output['random_state'] = param['random_state']

        #to research
        if 'max_leaf_nodes' in param:
            self.alg.set_params(max_leaf_nodes=param['max_leaf_nodes'])
            self.model_output['max_leaf_nodes'] = param['max_leaf_nodes']

        #whether to use Out of Bag samples for calculate generalization error
        if 'oob_score' in param:
            self.alg.set_params(oob_score=param['oob_score'])
            self.model_output['oob_score'] = param['oob_score']
        
        #cross validation folds
        if 'cv_folds' in param:
            self.cv_folds = param['cv_folds']

    #Fit the model using predictors and parameters specified before.
    def modelfit(self, printROC=False):

        #Outpute the parameters for the model for cross-checking:
        print 'Model being built with the following parameters:'
        print self.alg.get_params()

        self.alg.fit(self.data_train[self.predictors], self.data_train[self.target])
        
        print 'Feature Importance Scores: '
        featimp = pd.Series(self.alg.feature_importances_, index=self.predictors).sort_values()
        print featimp
        self.model_output['Feature_Importance'] = featimp.to_string()

        if self.model_output['oob_score']:
            print 'OOB Score : %f' % self.alg.oob_score_
            self.model_output['OOB_Score'] = self.alg.oob_score_

        #Get predictions:
        self.train_predictions = self.alg.predict(self.data_train[self.predictors])
        self.test_predictions_raw = self.alg.predict(self.data_test[self.predictors])
        self.test_predictions_transformed = self.output_transformation(self.test_predictions_raw)

        self.calc_model_characteristics()
        self.printReport()

    #Export the model into the model file as well as create a submission with model index. This will be used for creating an ensemble.
    def export_model(self, IDcol, submission_target_name=None):
        self.create_ensemble_dir()
        filename = os.path.join(os.getcwd(),'ensemble/rf_models.csv')
        comb_series = self.regression_output.append(self.model_output, verify_integrity=True)

        if os.path.exists(filename):
            models = pd.read_csv(filename)
            mID = int(max(models['ModelID'])+1)
        else:
            mID = 1
            models = pd.DataFrame(columns=comb_series.index)
            
        comb_series['ModelID'] = mID
        models = models.append(comb_series, ignore_index=True)
        
        models.to_csv(filename, index=False, float_format="%.5f")
        model_filename = os.path.join(os.getcwd(),'ensemble/rf_'+str(mID)+'.csv')
        self.submission(IDcol, model_filename, submission_target_name)

#######################################################################################################################
##### GRADIENT BOOSTING MACHINE
#######################################################################################################################

class GradientBoosting_Regr(GenericModelClass):
    def __init__(self,data_train, data_test, target, predictors=[],cv_folds=10):
        GenericModelClass.__init__(self, alg=GradientBoostingRegressor(), data_train=data_train, 
                                   data_test=data_test, target=target, predictors=predictors,cv_folds=cv_folds)
        self.default_parameters = {
                                    'loss':'ls', 'learning_rate':0.1, 'n_estimators':100, 'subsample':1.0, 'min_samples_split':2, 'min_samples_leaf':1,
                                    'max_depth':3, 'init':None, 'random_state':None, 'max_features':None, 'verbose':0, 'alpha':0.9,
                                    'max_leaf_nodes':None, 'warm_start':False, 'presort':'auto'
                                  }
        self.model_output = pd.Series(self.default_parameters)
        self.model_output['Feature_Importance'] = "-"
        
        #Set parameters to default values:
        self.set_parameters(set_default=True)

    # Set the parameters of the model. 
    # Note: 
    #     > only the parameters to be updated are required to be passed
    #     > if set_default is True, the passed parameters are ignored and default parameters are set which are defined in   scikit learn module
    def set_parameters(self, param=None, set_default=False):
        
        #Set param to default values if default set
        if set_default:
            param = self.default_parameters

        #Loss function to be used - deviance or exponential
        if 'loss' in param:
            self.alg.set_params(loss=param['loss'])
            self.model_output['loss'] = param['loss']
        
        if 'learning_rate' in param:
            self.alg.set_params(learning_rate=param['learning_rate'])
            self.model_output['learning_rate'] = param['learning_rate']
        
        #trees in forest:
        if 'n_estimators' in param:
            self.alg.set_params(n_estimators=param['n_estimators'])
            self.model_output['n_estimators'] = param['n_estimators']
        
        if 'subsample' in param:
            self.alg.set_params(subsample=param['subsample'])
            self.model_output['subsample'] = param['subsample']
        
        #maximum depth of each tree (ignored if max_leaf_nodes is not None)
        if 'max_depth' in param:
            self.alg.set_params(max_depth=param['max_depth'])
            self.model_output['max_depth'] = param['max_depth']
        
        #min #samples required to split an internal node; typically around 20-50
        if 'min_samples_split' in param:
            self.alg.set_params(min_samples_split=param['min_samples_split'])
            self.model_output['min_samples_split'] = param['min_samples_split']
        
        #The minimum number of samples in newly created leaves.
        if 'min_samples_leaf' in param:
            self.alg.set_params(min_samples_leaf=param['min_samples_leaf'])
            self.model_output['min_samples_leaf'] = param['min_samples_leaf']

        #max features to be considered for each split
        if 'max_features' in param:
            self.alg.set_params(max_features=param['max_features'])
            self.model_output['max_features'] = param['max_features']

        #for replication of results
        if 'random_state' in param:
            self.alg.set_params(random_state=param['random_state'])
            self.model_output['random_state'] = param['random_state']

        #to research
        if 'max_leaf_nodes' in param:
            self.alg.set_params(max_leaf_nodes=param['max_leaf_nodes'])
            self.model_output['max_leaf_nodes'] = param['max_leaf_nodes']

        #whether to use Out of Bag samples for calculate generalization error
        if 'presort' in param:
            self.alg.set_params(presort=param['presort'])
            self.model_output['presort'] = param['presort']

        if 'verbost' in param:  
            self.alg.set_params(verbose=param['verbose'])
            self.model_output['verbose'] = param['verbose']
        
        if 'warm_start' in param:
            self.alg.set_params(warm_start=param['warm_start'])
            self.model_output['warm_start'] = param['warm_start']

        if 'alpha' in param:
            self.alg.set_params(alpha=param['alpha'])
            self.model_output['alpha'] = param['alpha']
        
        #cross validation folds
        if 'cv_folds' in param:
            self.cv_folds = param['cv_folds']

    #Fit the model using predictors and parameters specified before.
    # Inputs:
    #     printROC - if True, prints an ROC curve. This functionality is currently not implemented
    def modelfit(self, performCV=True):

        #Outpute the parameters for the model for cross-checking:
        print 'Model being built with the following parameters:'
        print self.alg.get_params()

        self.alg.fit(self.data_train[self.predictors], self.data_train[self.target])
        
        print 'Feature Importance Scores: '
        self.feature_imp = pd.Series(self.alg.feature_importances_, index=self.predictors).sort_values()
        print self.feature_imp
        self.model_output['Feature_Importance'] = self.feature_imp.to_string()

        #Plot OOB estimates if subsample <1:
        if self.model_output['subsample']<1:
            plt.xlabel("GBM Iteration")
            plt.ylabel("Score")
            plt.plot(range(1, self.model_output['n_estimators']+1), self.alg.oob_improvement_)
            # plt.plot(range(1, self.model_output['n_estimators']+1), self.alg.train_score_)
            plt.legend(['oob_improvement_','train_score_'], loc='upper left')
            plt.show(block=False)

        #Get predictions:
        self.train_predictions = self.alg.predict(self.data_train[self.predictors])
        self.test_predictions_raw = self.alg.predict(self.data_test[self.predictors])
        self.test_predictions_transformed = self.output_transformation(self.test_predictions_raw)

        self.calc_model_characteristics()
        self.printReport()

    #Export the model into the model file as well as create a submission with model index. This will be used for creating an ensemble.
    def export_model(self, IDcol):
        self.create_ensemble_dir()
        filename = os.path.join(os.getcwd(),'ensemble/gbm_models.csv')
        comb_series = self.classification_output.append(self.model_output, verify_integrity=True)

        if os.path.exists(filename):
            models = pd.read_csv(filename)
            mID = int(max(models['ModelID'])+1)
        else:
            mID = 1
            models = pd.DataFrame(columns=comb_series.index)
            
        comb_series['ModelID'] = mID
        models = models.append(comb_series, ignore_index=True)
        
        models.to_csv(filename, index=False, float_format="%.5f")
        model_filename = os.path.join(os.getcwd(),'ensemble/gbm_'+str(mID)+'.csv')
        self.submission(IDcol, model_filename)

#######################################################################################################################
##### ENSEMBLE
#######################################################################################################################

#Class for creating an ensemble model using the exported files from previous classes

class Ensemble_Regression(object):
    #initialize the object with target variable
    def __init__(self, target):
        self.target = target
        self.data = None
        self.relationMatrix = None

    #create the ensemble data
    def create_ensemble_data(self, models, filename="Submission_ensemble.csv"):
        self.data = None
        for key, value in models.items():
            # print key,value
            for i in value:
                fname = key + '_' + str(i)
                fpath = os.path.join(os.getcwd(), 'ensemble', fname+'.csv')
                tempdata = pd.read_csv(fpath)
                tempdata = tempdata.rename(columns = {self.target: fname})
                if self.data is None:
                    self.data = tempdata
                else:
                    self.data = self.data.merge(tempdata,on=self.data.columns[0])
    
    #Create the relational matrix between models
    def check_corr(self):
        self.relationMatrix = self.data.corr()
        print self.relationMatrix

    #Generate submission for the ensembled model by combining the mentioned models.
    # Inputs:
    #     models_to_use - dictionary with key as the model name and values as list containing the model numbers to be ensebled
    #     filename - the filename of the final submission
    #     Note: the models should be odd in number to allow a clear winner in terms of mode otherwise the first element will be chosen 
    def submission(self, models_to_use=None, filename="Submission.csv"):

        #if models_to_use is None then use all, else filter:
        if models_to_use is None:
            data_ens = self.data
        else:
            data_ens = self.data[models_to_use]

        ensemble_output = data_ens.apply(np.mean,axis=1)
        submission = pd.DataFrame({
                IDcol: self.data.iloc[:,0],
                self.target: ensemble_output
            })
        submission.to_csv(filename, index=False)
