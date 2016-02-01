#######################################################################################################################
##### IMPORT STANDARD MODULES
#######################################################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydot
import os
from scipy.stats.mstats import chisquare, mode
        
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from sklearn.feature_selection import RFE, RFECV
from sklearn.externals.six import StringIO
import xgboost  

#######################################################################################################################
##### GENERIC MODEL CLASS
#######################################################################################################################

#This class contains the generic classification functions and variable definitions applicable across all models 
class GenericModelClass(object):
    def __init__(self, alg, data_train, data_test, target, predictors=[],cv_folds=10):
        self.alg = alg                  #an instance of particular model class
        self.data_train = data_train    #training data
        self.data_test = data_test      #testing data
        self.target = target
        self.cv_folds = cv_folds
        self.predictors = predictors
        self.train_predictions = []
        self.test_predictions = []
        self.test_pred_prob = []
        self.num_target_class = len(data_train[target].unique())

        #Define a Series object to store generic classification model outcomes; 
        self.classification_output=pd.Series(index=['ModelID','Accuracy','CVScore_mean','CVScore_std','AUC',
                                             'ActualScore (manual entry)','CVMethod','ConfusionMatrix','Predictors'])

        #not to be used for all but most
        self.feature_imp = None
    
    #Modify and get predictors for the model:
    def set_predictors(self, predictors):
        self.predictors=predictors

    def get_predictors(self):
        return self.predictors

    def get_test_predictions(self, getprob=False):
        if getprob:
            return self.test_pred_prob
        else:
            return self.test_predictions

    def get_feature_importance(self):
        return self.feature_imp

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
            error.append(self.alg.score(self.data_train[self.predictors].iloc[test,:], self.data_train[self.target].iloc[test]))
            
        return {'mean_error': np.mean(error),
                'std_error': np.std(error),
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
    def RecursiveFeatureEliminationCV(self, step=1, inplace=False, scoring='accuracy'):
        rfecv = RFECV(self.alg, step=step,cv=self.cv_folds,scoring=scoring)
        
        rfecv.fit(self.data_train[self.predictors], self.data_train[self.target])

        min_nfeat = len(self.predictors) - step*(len(rfecv.grid_scores_)-1)  # n - step*(number of iter - 1)
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(min_nfeat, len(self.predictors) + 1,step), rfecv.grid_scores_)
        plt.show(block=False)

        ranks = pd.Series(rfecv.ranking_, index=self.predictors)
        selected = ranks.loc[rfecv.support_]

        if inplace:
            self.set_predictors(selected.index.tolist())
        return ranks

    # Determine key metrics to analyze the classification model. These are stored in the classification_output series object belonginf to this class.
    def calc_model_characteristics(self, performCV=True):
        self.classification_output['Accuracy'] = metrics.accuracy_score(self.data_train[self.target],self.train_predictions)

        if performCV:
            cv_score= self.KFold_CrossValidation()
        else:
            cv_score={'mean_error': 0.0, 'std_error': 0.0}

        self.classification_output['CVMethod'] = 'KFold - ' + str(self.cv_folds)
        self.classification_output['CVScore_mean'] = cv_score['mean_error']
        self.classification_output['CVScore_std'] = cv_score['std_error']
        if self.num_target_class < 3:
            self.classification_output['AUC'] = metrics.roc_auc_score(self.data_train[self.target],self.test_pred_prob)
        else:
            self.classification_output['AUC'] = np.nan
        self.classification_output['ConfusionMatrix'] = pd.crosstab(self.data_train[self.target], self.train_predictions).to_string()
        self.classification_output['Predictors'] = str(self.predictors)

    # Print the metric determined in the previous function.
    def printReport(self, ROCcurve=False):
        print "\nModel Report"
        print "Confusion Matrix:"
        print pd.crosstab(self.data_train[self.target], self.train_predictions)
        print 'Note: rows - actual; col - predicted'
        # print "\nClassification Report:"
        # print metrics.classification_report(y_true=self.data_train[self.target], y_pred=self.train_predictions)
        print "Accuracy : %s" % "{0:.3%}".format(self.classification_output['Accuracy'])
        print "AUC : %s" % "{0:.3%}".format(self.classification_output['AUC'])
        print "CV Score : Mean - %s | Std - %s" % ("{0:.3%}".format(self.classification_output['CVScore_mean']),"{0:.3%}".format(self.classification_output['CVScore_std']))
        
        if ROCcurve:
            fpr, tpr, thresholds = metrics.roc_curve(self.data_train[self.target],self.test_pred_prob)

    # create submission file
    def submission(self, IDcol, filename="Submission.csv"):
        submission = pd.DataFrame({
                IDcol: self.data_test[IDcol],
                self.target: self.test_predictions.astype(int)
            })
        submission.to_csv(filename, index=False)

    #checks whether the ensemble directory exists and creates one if it doesn't
    def create_ensemble_dir(self):
        ensdir = os.path.join(os.getcwd(), 'ensemble')
        if not os.path.isdir(ensdir):
            os.mkdir(ensdir)

#######################################################################################################################
##### LOGISTIC REGRESSION
#######################################################################################################################

class Logistic_Regression(GenericModelClass):

    def __init__(self,data_train, data_test, target, predictors=[],cv_folds=10):
        GenericModelClass.__init__(self, alg=LogisticRegression(), data_train=data_train, data_test=data_test, target=target, predictors=predictors,cv_folds=cv_folds)
        self.default_parameters = {'C':1.0, 'tol':0.0001, 'solver':'liblinear','multi_class':'ovr','class_weight':'balanced'}
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

        if 'C' in param:
            self.alg.set_params(C=param['C'])
            self.model_output['C'] = param['C']

        if 'tol' in param:
            self.alg.set_params(tol=param['tol'])
            self.model_output['tol'] = param['tol']

        if 'solver' in param:
            self.alg.set_params(solver=param['solver'])
            self.model_output['solver'] = param['solver']

        if 'multi_class' in param:
            self.alg.set_params(multi_class=param['multi_class'])
            self.model_output['multi_class'] = param['multi_class']

        if 'class_weight' in param:
            self.alg.set_params(class_weight=param['class_weight'])
            self.model_output['class_weight'] = param['class_weight']

        if 'cv_folds' in param:
            self.cv_folds = param['cv_folds']
    
    #Fit the model using predictors and parameters specified before.
    # Inputs:
    #     printROC - if True, prints an ROC curve. This functionality is currently not implemented
    def modelfit(self, printROC=False, performCV=True):

        #Outpute the parameters for the model for cross-checking:
        print 'Model being built with the following parameters:'
        print self.alg.get_params()
        
        self.alg.fit(self.data_train[self.predictors], self.data_train[self.target])
        coeff = pd.Series(np.concatenate((self.alg.intercept_,self.alg.coef_[0,:])), index=["Intercept"]+self.predictors)
        print 'Coefficients: '
        print coeff

        self.model_output['Coefficients'] = coeff.to_string()
        
        #Get predictions:
        self.test_predictions = self.alg.predict(self.data_test[self.predictors])
        self.train_predictions = self.alg.predict(self.data_train[self.predictors])
        self.test_pred_prob = self.alg.predict_proba(self.data_train[self.predictors])[:,1]

        self.calc_model_characteristics(performCV)
        self.printReport(ROCcurve=printROC)
    
    #Export the model into the model file as well as create a submission with model index. This will be used for creating an ensemble.
    def export_model(self, IDcol):
        self.create_ensemble_dir()
        filename = os.path.join(os.getcwd(),'ensemble/logreg_models.csv')
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
        model_filename = os.path.join(os.getcwd(),'ensemble/logreg_'+str(mID)+'.csv')
        self.submission(IDcol, model_filename)


#######################################################################################################################
##### DECISION TREE
#######################################################################################################################

class Decision_Tree_Class(GenericModelClass):
    def __init__(self,data_train, data_test, target, predictors=[],cv_folds=10):
        GenericModelClass.__init__(self, alg=DecisionTreeClassifier(), data_train=data_train, 
                                   data_test=data_test, target=target, predictors=predictors,cv_folds=cv_folds)
        self.default_parameters = {'criterion':'gini', 'max_depth':None, 
                                   'min_samples_split':2, 'min_samples_leaf':1, 
                                   'max_features':None, 'random_state':None, 'max_leaf_nodes':None, 'class_weight':'balanced'}
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

        if 'class_weight' in param:
            self.alg.set_params(class_weight=param['class_weight'])
            self.model_output['class_weight'] = param['class_weight']

        if 'cv_folds' in param:
            self.cv_folds = param['cv_folds']
    
    #Fit the model using predictors and parameters specified before.
    # Inputs:
    #     printROC - if True, prints an ROC curve. This functionality is currently not implemented
    def modelfit(self, printROC=False, performCV=True):

        #Outpute the parameters for the model for cross-checking:
        print 'Model being built with the following parameters:'
        print self.alg.get_params()
        
        self.alg.fit(self.data_train[self.predictors], self.data_train[self.target])
        
        print 'Feature Importance Scores: '
        self.feature_imp = pd.Series(self.alg.feature_importances_, index=self.predictors).sort_values()
        print self.feature_imp
        self.model_output['Feature_Importance'] = self.feature_imp.to_string()

        #Get predictions:
        self.test_predictions = self.alg.predict(self.data_test[self.predictors])
        self.train_predictions = self.alg.predict(self.data_train[self.predictors])
        self.test_pred_prob = self.alg.predict_proba(self.data_test[self.predictors])
        # print self.test_pred_prob
        # print self.test_predictions

        self.calc_model_characteristics(performCV)
        self.printReport(ROCcurve=printROC)

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
    def export_model(self, IDcol):
        self.create_ensemble_dir()
        filename = os.path.join(os.getcwd(),'ensemble/dectree_models.csv')
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
        model_filename = os.path.join(os.getcwd(),'ensemble/dectree_'+str(mID)+'.csv')
        self.submission(IDcol, model_filename)

#######################################################################################################################
##### RANDOM FOREST
#######################################################################################################################

class Random_Forest_Class(GenericModelClass):
    def __init__(self,data_train, data_test, target, predictors=[],cv_folds=10):
        GenericModelClass.__init__(self, alg=RandomForestClassifier(), data_train=data_train, 
                                   data_test=data_test, target=target, predictors=predictors,cv_folds=cv_folds)
        self.default_parameters = {
                                    'n_estimators':10, 'criterion':'gini', 'max_depth':None,' min_samples_split':2, 
                                    'min_samples_leaf':1, 'max_features':'auto', 'max_leaf_nodes':None,
                                    'oob_score':False, 'random_state':None, 'class_weight':'balanced' 
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

        if 'class_weight' in param:
            self.alg.set_params(class_weight=param['class_weight'])
            self.model_output['class_weight'] = param['class_weight']

        #cross validation folds
        if 'cv_folds' in param:
            self.cv_folds = param['cv_folds']

    #Fit the model using predictors and parameters specified before.
    # Inputs:
    #     printROC - if True, prints an ROC curve. This functionality is currently not implemented
    def modelfit(self, printROC=False, performCV=True):

        #Outpute the parameters for the model for cross-checking:
        print 'Model being built with the following parameters:'
        print self.alg.get_params()
        
        self.alg.fit(self.data_train[self.predictors], self.data_train[self.target])
        
        print 'Feature Importance Scores: '
        self.feature_imp = pd.Series(self.alg.feature_importances_, index=self.predictors).sort_values()
        print self.feature_imp
        self.model_output['Feature_Importance'] = self.feature_imp.to_string()

        if self.model_output['oob_score']:
            print 'OOB Score : %f' % self.alg.oob_score_
            self.model_output['OOB_Score'] = self.alg.oob_score_

        #Get predictions:
        self.test_predictions = self.alg.predict(self.data_test[self.predictors])
        self.train_predictions = self.alg.predict(self.data_train[self.predictors])
        self.test_pred_prob = self.alg.predict_proba(self.data_train[self.predictors])[:,1]

        self.calc_model_characteristics(performCV)
        self.printReport(ROCcurve=printROC)

    #Export the model into the model file as well as create a submission with model index. This will be used for creating an ensemble.
    def export_model(self, IDcol):
        self.create_ensemble_dir()
        filename = os.path.join(os.getcwd(),'ensemble/rf_models.csv')
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
        model_filename = os.path.join(os.getcwd(),'ensemble/rf_'+str(mID)+'.csv')
        self.submission(IDcol, model_filename)

#######################################################################################################################
##### ADABOOST CLASSIFICATION
#######################################################################################################################

class AdaBoost_Class(GenericModelClass):
    def __init__(self,data_train, data_test, target, predictors=[],cv_folds=10):
        GenericModelClass.__init__(self, alg=AdaBoostClassifier(), data_train=data_train, 
                                   data_test=data_test, target=target, predictors=predictors,cv_folds=cv_folds)
        self.default_parameters = { 'n_estimators':50, 'learning_rate':1.0 }
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

        #trees in forest:
        if 'n_estimators' in param:
            self.alg.set_params(n_estimators=param['n_estimators'])
            self.model_output['n_estimators'] = param['n_estimators']
        
        #decision tree split criteria
        if 'learning_rate' in param:
            self.alg.set_params(learning_rate=param['learning_rate'])
            self.model_output['learning_rate'] = param['learning_rate']
        
        if 'cv_folds' in param:
            self.cv_folds = param['cv_folds']

    #Fit the model using predictors and parameters specified before.
    # Inputs:
    #     printROC - if True, prints an ROC curve. This functionality is currently not implemented
    def modelfit(self, printROC=False, performCV=True):

        #Outpute the parameters for the model for cross-checking:
        print 'Model being built with the following parameters:'
        print self.alg.get_params()
        
        self.alg.fit(self.data_train[self.predictors], self.data_train[self.target])
        
        print 'Feature Importance Scores: '
        self.feature_imp = pd.Series(self.alg.feature_importances_, index=self.predictors).sort_values()
        print self.feature_imp
        self.model_output['Feature_Importance'] = self.feature_imp.to_string()

        plt.xlabel("AdaBoost Estimator")
        plt.ylabel("Estimator Error")
        plt.plot(range(1, self.model_output['n_estimators']+1), self.alg.estimator_errors_)
        plt.plot(range(1, self.model_output['n_estimators']+1), self.alg.estimator_weights_)
        plt.legend(['estimator_errors','estimator_weights'], loc='upper left')
        plt.show(block=False)

        #Get predictions:
        self.test_predictions = self.alg.predict(self.data_test[self.predictors])
        self.train_predictions = self.alg.predict(self.data_train[self.predictors])
        self.test_pred_prob = self.alg.predict_proba(self.data_train[self.predictors])[:,1]

        self.calc_model_characteristics(performCV)
        self.printReport(ROCcurve=printROC)

    #Export the model into the model file as well as create a submission with model index. This will be used for creating an ensemble.
    def export_model(self, IDcol):
        self.create_ensemble_dir()
        filename = os.path.join(os.getcwd(),'ensemble/adaboost_models.csv')
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
        model_filename = os.path.join(os.getcwd(),'ensemble/adaboost_'+str(mID)+'.csv')
        self.submission(IDcol, model_filename)

#######################################################################################################################
##### GRADIENT BOOSTING MACHINE
#######################################################################################################################

class GradientBoosting_Class(GenericModelClass):
    def __init__(self,data_train, data_test, target, predictors=[],cv_folds=10):
        GenericModelClass.__init__(self, alg=GradientBoostingClassifier(), data_train=data_train, 
                                   data_test=data_test, target=target, predictors=predictors,cv_folds=cv_folds)
        self.default_parameters = {
                                    'loss':'deviance', 'learning_rate':0.1, 'n_estimators':100, 'subsample':1.0, 'min_samples_split':2, 'min_samples_leaf':1,
                                    'max_depth':3, 'init':None, 'random_state':None, 'max_features':None, 'verbose':0, 
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
        
        #cross validation folds
        if 'cv_folds' in param:
            self.cv_folds = param['cv_folds']

    #Fit the model using predictors and parameters specified before.
    # Inputs:
    #     printROC - if True, prints an ROC curve. This functionality is currently not implemented
    def modelfit(self, printROC=False, performCV=True):

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
        self.test_predictions = self.alg.predict(self.data_test[self.predictors])
        self.train_predictions = self.alg.predict(self.data_train[self.predictors])
        self.test_pred_prob = self.alg.predict_proba(self.data_train[self.predictors])[:,1]

        self.calc_model_characteristics(performCV)
        self.printReport(ROCcurve=printROC)

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
##### XGBOOST ALGORITHM
#######################################################################################################################

class XGBoost_Class(GenericModelClass):
    def __init__(self,data_train, data_test, target, predictors=[],cv_folds=10):
        GenericModelClass.__init__(self, alg=xgboost.XGBClassifier(), data_train=data_train, 
                                   data_test=data_test, target=target, predictors=predictors,cv_folds=cv_folds)
        self.default_parameters = { 
                                 'max_depth':3, 'learning_rate':0.1,
                                 'n_estimators':100, 'silent':True,
                                 'objective':"binary:logistic",
                                 'nthread':-1, 'gamma':0, 'min_child_weight':1,
                                 'max_delta_step':0, 'subsample':1, 'colsample_bytree':1, 'colsample_bylevel':1,
                                 'reg_alpha':0, 'reg_lambda':1, 'scale_pos_weight':1,
                                 'base_score':0.5, 'seed':0, 'missing':None
                            }
        self.model_output = pd.Series(self.default_parameters)
        self.model_output['Feature_Importance'] = "-"

        self.eval_metric = None  #To update with set parameters

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
        if 'max_depth' in param:
            self.alg.set_params(max_depth=param['max_depth'])
            self.model_output['max_depth'] = param['max_depth']

        if 'learning_rate' in param:
            self.alg.set_params(learning_rate=param['learning_rate'])
            self.model_output['learning_rate'] = param['learning_rate']

        if 'n_estimators' in param:
            self.alg.set_params(n_estimators=param['n_estimators'])
            self.model_output['n_estimators'] = param['n_estimators']

        if 'silent' in param:
            self.alg.set_params(silent=param['silent'])
            self.model_output['silent'] = param['silent']

        if 'objective' in param:
            self.alg.set_params(objective=param['objective'])
            self.model_output['objective'] = param['objective']

        if 'gamma' in param:
            self.alg.set_params(gamma=param['gamma'])
            self.model_output['gamma'] = param['gamma']

        if 'min_child_weight' in param:
            self.alg.set_params(min_child_weight=param['min_child_weight'])
            self.model_output['min_child_weight'] = param['min_child_weight']

        if 'max_delta_step' in param:
            self.alg.set_params(max_delta_step=param['max_delta_step'])
            self.model_output['max_delta_step'] = param['max_delta_step']

        if 'subsample' in param:
            self.alg.set_params(subsample=param['subsample'])
            self.model_output['subsample'] = param['subsample']

        if 'colsample_bytree' in param:
            self.alg.set_params(colsample_bytree=param['colsample_bytree'])
            self.model_output['colsample_bytree'] = param['colsample_bytree']

        if 'eval_metric' in param:
            self.eval_metric = param['eval_metric']
    

        # if '' in param:
        #     self.alg.set_params(=param[''])
        #     self.model_output[''] = param['']

        # if '' in param:
        #     self.alg.set_params(=param[''])
        #     self.model_output[''] = param['']
    
    def KFold_CrossValidation(self):
        # print 'xgboost override working'
        # Generate cross validation folds for the training dataset.  
        kf = KFold(self.data_train.shape[0], n_folds=self.cv_folds)
        
        error = []
        for train, test in kf:
            # Filter training data
            train_predictors = (self.data_train[self.predictors].iloc[train,:])
            # The target we're using to train the algorithm.
            train_target = self.data_train[self.target].iloc[train]
            # Training the algorithm using the predictors and target.
            self.alg.fit(train_predictors, train_target, eval_metric=self.eval_metric)
            
            test_predictors = (self.data_train[self.predictors].iloc[test,:])
            test_target = self.data_train[self.target].iloc[test]

            #Record error from each cross-validation run
            error.append(metrics.accuracy_score(test_target, self.alg.predict(test_predictors)))
            
        return {'mean_error': np.mean(error),
                'std_error': np.std(error),
                'all_error': error }

    #Fit the model using predictors and parameters specified before.
    # Inputs:
    #     printROC - if True, prints an ROC curve. This functionality is currently not implemented
    def modelfit(self, printROC=False, performCV=True):
        self.alg.fit(self.data_train[self.predictors], self.data_train[self.target], eval_metric=self.eval_metric)
        
        # print self.alg.evals_result()
        print 'Feature Importance Scores: '
        self.feature_imp = pd.Series(self.alg.booster().get_fscore(), index=self.predictors).sort_values()
        print self.feature_imp
        self.model_output['Feature_Importance'] = self.feature_imp.to_string()

        #Get predictions:
        self.test_predictions = self.alg.predict(self.data_test[self.predictors])
        self.train_predictions = self.alg.predict(self.data_train[self.predictors])
        self.test_pred_prob = self.alg.predict_proba(self.data_train[self.predictors])[:,1]

        self.calc_model_characteristics(performCV)
        self.printReport(ROCcurve=printROC)

    #Export the model into the model file as well as create a submission with model index. This will be used for creating an ensemble.
    def export_model(self, IDcol):
        self.create_ensemble_dir()
        filename = os.path.join(os.getcwd(),'ensemble/xgboost_models.csv')
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
        model_filename = os.path.join(os.getcwd(),'ensemble/xgboost_'+str(mID)+'.csv')
        self.submission(IDcol, model_filename)

#######################################################################################################################
##### ENSEMBLE
#######################################################################################################################

#Class for creating an ensemble model using the exported files from previous classes
class Ensemble_Classification(object):
    #initialize the object with target variable
    def __init__(self, target, IDcol):
        self.target = target
        self.data = None
        self.relationMatrix_chi2 = None
        self.relationMatrix_diff = None
        self.IDcol = IDcol

    #create the ensemble data
    # Inputs:
    #     models - dictionary with key as the model name and values as list containing the model numbers to be ensebled
    # Note: all the models in the list specified should be present in the ensemble folder. Please cross-check once 
    def create_ensemble_data(self, models):
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

    #get the data being used for ensemble
    def get_ensemble_data(self):
        return self.data
    
    #Check chisq test between different model outputs to check which combination of ensemble will generate better results. Note: Models with high correlation should not be combined together.
    def chisq_independence(self, col1, col2, verbose = False):
        contingencyTable = pd.crosstab(col1,col2,margins=True)

        if len(col1)/((contingencyTable.shape[0] - 1) * (contingencyTable.shape[1] - 1)) <= 5:
            return "TMC"

        expected = contingencyTable.copy()
        total = contingencyTable.loc["All","All"]
        # print contingencyTable.index
        # print contingencyTable.columns
        for m in contingencyTable.index:
            for n in contingencyTable.columns:
                expected.loc[m,n] = contingencyTable.loc[m,"All"]*contingencyTable.loc["All",n]/float(total)
        
        if verbose:
            print '\n\nAnalysis of models: %s and %s' % (col1.name, col2.name)
            print 'Contingency Table:'
            print contingencyTable
            # print '\nExpected Frequency Table:'
            # print expected
        observed_frq = contingencyTable.iloc[:-1,:-1].values.ravel()
        expected_frq = expected.iloc[:-1,:-1].values.ravel()

        numless1 = len(expected_frq[expected_frq<1])
        perless5 = len(expected_frq[expected_frq<5])/len(expected_frq)

        #Adjustment in DOF so use the 1D chisquare to matrix shaped data; -1 in row n col because of All row and column
        matrixadj = (contingencyTable.shape[0] - 1) + (contingencyTable.shape[1] - 1) - 2
        # print matrixadj
        pval = np.round(chisquare(observed_frq, expected_frq,ddof=matrixadj)[1],3)

        if numless1>0 or perless5>=0.2:
            return str(pval)+"*"
        else: 
            return pval

    #Create the relational matrix between models
    def check_ch2(self, verbose=False):
        col = self.data.columns[1:]
        self.relationMatrix_chi2 = pd.DataFrame(index=col,columns=col)

        for i in range(len(col)):
            for j in range(i, len(col)):
                if i==j:
                    self.relationMatrix_chi2.loc[col[i],col[j]] = 1
                else:
                    pval = self.chisq_independence(self.data.iloc[:,i+1],self.data.iloc[:,j+1], verbose=verbose)
                    self.relationMatrix_chi2.loc[col[j],col[i]] = pval
                    self.relationMatrix_chi2.loc[col[i],col[j]] = pval

        print '\n\n Relational Matrix (based on Chi-square test):'
        print self.relationMatrix_chi2

    def check_diff(self):
        col = self.data.columns[1:]
        self.relationMatrix_diff = pd.DataFrame(index=col,columns=col)
        nrow = self.data.shape[0]
        for i in range(len(col)):
            for j in range(i, len(col)):
                if i==j:
                    self.relationMatrix_diff.loc[col[i],col[j]] = '-'
                else:
                    # print col[i],col[j]
                    pval = "{0:.2%}".format(sum( np.abs(self.data.iloc[:,i+1]-self.data.iloc[:,j+1]) )/float(nrow))
                    self.relationMatrix_diff.loc[col[j],col[i]] = pval
                    self.relationMatrix_diff.loc[col[i],col[j]] = pval

        print '\n\n Relational Matrix (based on perc difference):'
        print self.relationMatrix_diff


    #Generate submission for the ensembled model by combining the mentioned models.
    # Inputs:
    #     models_to_use - list with model names to use; if None- all models will be used
    #     filename - the filename of the final submission
    #     Note: the models should be odd in nucmber to allow a clear winner in terms of mode otherwise the first element will be chosen 
    def submission(self, models_to_use=None, filename="Submission_ensemble.csv"):

        #if models_to_use is None then use all, else filter:
        if models_to_use is None:
            data_ens = self.data
        else:
            data_ens = self.data[models_to_use]

        def mode_ens(x):
            return int(mode(x).mode[0])

        ensemble_output = data_ens.apply(mode_ens,axis=1)
        submission = pd.DataFrame({
                self.IDcol: self.data.iloc[:,0],
                self.target: ensemble_output
            })
        submission.to_csv(filename, index=False)