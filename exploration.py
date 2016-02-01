###############################################################################
##### SUMMARY (HELP)
###############################################################################

"""
Function: exploration(data_train, data_test, outcome)
    used to create an excel file containing basic information about the data passed as arguments
    Arguments: 
        a. data_train - Pandas dataframe containing the training data 
        b. data_test - Pandas dataframe containing the testing data 
        c. outcome - a string constant specifying the name of the outcome variable
    
    Additional Requirements: 
        a. a datatypes.csv file in current folder specifying the datatype of each column
    
    Output: 
        a. DataAnalysis.xlsx - contains detailed report of data
"""

###############################################################################
##### IMPORT STANDARD MODULES
###############################################################################

import pandas as pd
import numpy as np
import os
import matplotlib as plt
from sklearn import feature_selection
from scipy.stats.mstats import chisquare
from openpyxl import load_workbook
import shutil 

#Note: xlrd should be installed for read_excel to work

from .setup import module_location

###############################################################################
##### DEFINE CLASSES
###############################################################################
    
#class for categorical variable
class categorical(object):
    def __init__(self, column):
        #self.data = data
        self.column = column
        self.frq = column.value_counts()
    
    #default print method:
    def __str__(self):
        print self.frq
        
    #Univariate analysis:
    def unique(self):
        return len(self.frq)
        
    def missing(self):
        return (self.column.size - sum(self.frq))
        
    def top5(self):
        outstr = ""
        for x in self.frq.keys()[:5]:
            if x:
                outstr += " | "
            outstr += "%s (%d:%s)" % (str(x),self.frq[x],"{0:.0%}".format(float(self.frq[x])/sum(self.frq)))
        return outstr
            
    def concentration(self):
        cumfrq = []
        cumperc = []
        out = [0,0,0,0]
        total = float(sum(self.frq))
        check = [True,True,True,True]
        
        #Following code can be made more efficient by keeping only current values
        for i in range(len(self.frq)):
            x = self.frq.iloc[i]
            if i==0:
                cumfrq.append(x)
                cumperc.append(x/total)
            else:
                t = x+cumfrq[i-1]
                cumfrq.append(t)
                cumperc.append(t/total)        
            #print cumfrq, cumperc,out
            for c in range(4):
                if check[c]:
                    #print cumperc[i]
                    if cumperc[i] > (0.2*(c+1)):
                        out[c] = i+1
                        check[c] = False
        
        #print self.frq
        #print cumfrq, cumperc, out
        return out
                    
#class for categorical variable
class continuous(object):
    def __init__(self, column):
        self.column = column
        self.stats = column.describe()
        
    #default print method:
    def __str__(self):
        print self.data
        
    #Univariate analysis:
    def missing(self):
        return (len(self.column)-self.stats["count"])
        
    def statistics(self):
        return np.round(self.stats[1:].tolist(),2)
        
    def ValuesBeyondRange(self):
        iqr = self.stats["75%"] - self.stats["25%"]
        def check_outlier(x,iqr):
            if ((x < (self.stats["25%"] - 1.5*iqr))|( x > (self.stats["75%"] + 1.5*iqr))):
                return True
            else:
                return False
        return sum(self.column.apply(check_outlier, args=(iqr,)))
        
###############################################################################
##### DEFINE SUPPORT FUNCTIONS
###############################################################################

#Define a function to check datatype of column:
def check_type(coltype):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    
    if coltype in numerics:
        return "continuous"
    else:
        return "categorical"

def categorical_summary(data, ncol, relationMatrix, outcome):
    #Create Output DataFrame for categorical data:
    table_cols = ["Feature","#unique_values","#missing_values","Significance","20%_conc","40%_conc","60%_conc","80%_conc","Top5_Categories"]
    summary = pd.DataFrame(index=range(ncol), 
                           columns=[table_cols])
    
    i=0
    for col in data.columns:
        class_cat = categorical(data[col])
        summary.loc[i,table_cols[0]] = col
        summary.loc[i,table_cols[1]] = class_cat.unique()
        summary.loc[i,table_cols[2]] = class_cat.missing()
        if outcome:
            summary.loc[i,table_cols[3]] = relationMatrix.loc[outcome, col]
        else:
            summary.loc[i,table_cols[3]] = "-"
        summary.loc[i,table_cols[4:8]] = class_cat.concentration()
        summary.loc[i,table_cols[8]] = class_cat.top5()
        i+=1
        
    return summary    
    
def continuous_summary(data, ncol, relationMatrix, outcome):
    #Create Output DataFrame for continuous data:
    table_cols = ["Feature","#missing_values","Significance","Mean","Std","Min","25%","Median","75%","Max","#values_beyond_1.5IQR"]
    summary = pd.DataFrame(index=range(ncol), columns=table_cols)
    
    i=0
    for col in data.columns:
        class_cont = continuous(data[col])
        summary.loc[i,table_cols[0]] = col
        summary.loc[i,table_cols[1]] = class_cont.missing()
        if outcome:
            summary.loc[i,table_cols[2]] = relationMatrix.loc[outcome, col]
        else:
            summary.loc[i,table_cols[2]] = "-"
        summary.loc[i,table_cols[3:10]] = class_cont.statistics()
        summary.loc[i,table_cols[10]] = class_cont.ValuesBeyondRange()
        i+=1
        
    return summary

def chisq_independence(col1,col2):
    # print col1, col2
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
    
    # print contingencyTable
    # print expected
    observed_frq = contingencyTable.iloc[:-1,:-1].values.ravel()
    expected_frq = expected.iloc[:-1,:-1].values.ravel()

    numless1 = len(expected_frq[expected_frq<1])
    perless5 = len(expected_frq[expected_frq<5])/len(expected_frq)

    #Adjustment in DOF so use the 1D chisquare to matrix shaped data; -1 in row n col because of All row and column
    matrixadj = (contingencyTable.shape[0] - 1) + (contingencyTable.shape[1] - 1) - 2
    pval = np.round(chisquare(observed_frq, expected_frq,ddof=matrixadj)[1],3)

    if numless1>0 or perless5>=0.2:
        return str(pval)+"*"
    else: 
        return pval

#Function to perform chi-square/anova/correlation depending upon input
def SignificanceMatrix(data):
    col = data.columns
    colTypes = [ check_type(x) for x in data.dtypes ]
    relationMatrix = pd.DataFrame(index=col,columns=col)

    for i in range(len(col)):
        for j in range(i, len(col)):
            if i==j:
                pval = 1
                relationMatrix.loc[col[i],col[j]] = pval
            else:
                tempdata = data[[col[i],col[j]]]
                tempdata = tempdata.dropna(axis=0)   #Remeber to add warning where missing data is removed
                col1 = tempdata[col[i]]
                col2 = tempdata[col[j]].ravel()
                # print tempdata.dtypes
                # print colTypes[i],colTypes[j]
                if colTypes[i] == colTypes[j]:
                    if colTypes[i] == "continuous":
                        # print "both cont"
                        pval = np.round(feature_selection.f_regression(pd.DataFrame(col1),col2)[1][0],3)
                    else:
                        pval = chisq_independence(tempdata[col[i]],tempdata[col[j]])                        
                else:
                    if colTypes[i] == "continuous":
                        pval = np.round(feature_selection.f_classif(pd.DataFrame(col1),col2)[1][0],3)
                    else:
                        pval = np.round(feature_selection.f_classif(pd.DataFrame(col2),col1)[1][0],3)
                relationMatrix.loc[col[i],col[j]] = pval
                relationMatrix.loc[col[j],col[i]] = pval

    return relationMatrix.fillna("NAN")

def export_to_excel(data, sheetname, row_offset, col_offset):
    book = load_workbook("DataAnalysis.xlsx")
    writer = pd.ExcelWriter("DataAnalysis.xlsx", engine='openpyxl')
    writer.book = book
    writer.sheets = dict((ws.title,ws) for ws in book.worksheets)
    data.to_excel(writer,sheet_name=sheetname, startrow=row_offset, startcol=col_offset)
    writer.save()

###############################################################################
##### DEFINE EXPLORATION FUNCTION
###############################################################################
        
def exploration(data_train, data_test, outcome, verbose=True):
    
    #Print status if verbose true:
    if verbose:
        print 'Creating Overall Summary.... ',
    #load the data type from class:
    datatypes = pd.read_csv("datatypes.csv")
    # print datatypes.ix[:,1]
    col_categorical = datatypes.loc[datatypes.ix[:,1]=="categorical",datatypes.columns[0]].values
    col_continuous = datatypes.loc[datatypes.ix[:,1]=="continuous",datatypes.columns[0]].values

    #Check which columns are there in train data:
    col_categorical_train = [x for x in col_categorical if x in data_train.columns]
    col_continuous_train = [x for x in col_continuous if x in data_train.columns]
    
    #Load Training Data
    data_train_categorical = pd.DataFrame(data_train,columns=col_categorical_train,dtype=np.object)
    data_train_continuous = pd.DataFrame(data_train,columns=col_continuous_train,dtype=np.float)

    # print data_train_categorical.dtypes
    # print data_train_continuous.dtypes
    shape_train = data_train.shape
    shape_train_categorical = data_train_categorical.shape
    shape_train_continuous = data_train_continuous.shape
    
    #Define variable to adjust for outcome variable:
    #Currently this functionality is disables but just keeping the code in case change needed later. To be removed in final version
    if outcome in data_train_categorical.columns:
        catadj = 0
        conadj = 0
    else:
        catadj = 0
        conadj = 0
    
    #Check which columns are there in test data:
    col_categorical_test = [x for x in col_categorical if x in data_test.columns]
    col_continuous_test = [x for x in col_continuous if x in data_test.columns]
    
    #Load Test Data
    data_test_categorical = pd.DataFrame(data_test,columns=col_categorical_test,dtype=np.object)
    data_test_continuous = pd.DataFrame(data_test,columns=col_continuous_test,dtype=np.float)
    
    shape_test = data_test.shape
    shape_test_categorical = data_test_categorical.shape
    shape_test_continuous = data_test_continuous.shape    
    
    #Create combined dataset without outcome column:
    data_combined_categorical = pd.concat([data_train_categorical, data_test_categorical]).drop(outcome,axis=1,errors="ignore")
    data_combined_continuous = pd.concat([data_train_continuous, data_test_continuous]).drop(outcome,axis=1,errors="ignore")

    shape_combined_categorical = data_combined_categorical.shape
    shape_combined_continuous = data_combined_continuous.shape

    # print shape_test_continuous
    # print shape_test_categorical

    #Create Summary table:
    OverallSummary = pd.DataFrame({"Property": ["#Features","#Records","#Categorical Features","#Continuous Features"],
                 "Train_Value":[shape_train[1],shape_train[0],shape_train_categorical[1]-catadj,shape_train_continuous[1]-conadj],
                 "Test_Value":[shape_test[1],shape_test[0],shape_test_categorical[1],shape_test_continuous[1]] },
                 columns = ["Property","Train_Value","Test_Value"])
    
    #Print status if verbose true:
    if verbose:
        print ' Complete!'
        print OverallSummary
        print 'Creating Relational Matrices.... ',

    # print data_train_categorical
    # Print training correlation matrix:
    RelationalMatrix_train = SignificanceMatrix(pd.concat([data_train_categorical,data_train_continuous],axis=1))
    # print RelationalMatrix_train
    
    # Print testing correlation matrix:
    # print pd.concat([data_test_categorical,data_test_continuous],axis=1).head()
    RelationalMatrix_test = SignificanceMatrix(pd.concat([data_test_categorical,data_test_continuous],axis=1))
    # print RelationalMatrix
    
    # Print combined correlation matrix:
    RelationalMatrix_combined = SignificanceMatrix(pd.concat([data_combined_categorical,data_combined_continuous],axis=1))
    # print RelationalMatrix
    
    #Print status if verbose true:
    if verbose:
        print ' Complete!'
        print 'Creating Summary of Categorical Variables.... ',

    #Create training set categorical summary:
    summary_train_categorical = categorical_summary(data_train_categorical, shape_train_categorical[1], RelationalMatrix_train, outcome)
    # print summary_train_categorical 

    #Create testing set categorical summary:
    summary_test_categorical = categorical_summary(data_test_categorical, shape_test_categorical[1], RelationalMatrix_test, None)
    # print summary_test_categorical 

    #Create combined set categorical summary
    summary_combined_categorical = categorical_summary(data_combined_categorical, shape_combined_categorical[1], RelationalMatrix_combined, None)
    # print summary_test_categorical 

    #Print status if verbose true:
    if verbose:
        print ' Complete!'
        print 'Creating Summary of Continuous Variables.... ',

    #Create training set continuous summary:
    summary_train_continuous = continuous_summary(data_train_continuous, shape_train_continuous[1],RelationalMatrix_train, outcome)
    # print summary_train_continuous
    
    #Create training set continuous summary:
    summary_test_continuous = continuous_summary(data_test_continuous, shape_test_continuous[1],RelationalMatrix_test, None)
    # print summary_test_continuous

    #Create combined set continuous summary
    summary_combined_continuous = continuous_summary(data_combined_continuous, shape_combined_continuous[1],RelationalMatrix_combined, None)
    # print summary_test_categorical     

    # Feature Summary
    feature_summary = pd.DataFrame(index=range(shape_train_categorical[1]+shape_train_continuous[1]),columns=["Feature","Type","#unique(Train)","#unique(Test)"])

    for i in summary_train_categorical.index:
        feature_summary.loc[i,"Feature"] = summary_train_categorical.loc[i,"Feature"]
        feature_summary.loc[i,"Type"] = "Categorical"
        feature_summary.loc[i,"#unique(Train)"] = summary_train_categorical.loc[i,"#unique_values"]
        if summary_train_categorical.loc[i,"Feature"]==outcome:
            feature_summary.loc[i,"#unique(Test)"] = "-"
        else:    
            feature_summary.loc[i,"#unique(Test)"] = summary_test_categorical.loc[summary_test_categorical["Feature"]==summary_train_categorical.loc[i,"Feature"],"#unique_values"].values[0]
        
    numcat = shape_train_categorical[1]
    for i in summary_train_continuous.index:
        feature_summary.loc[i+numcat,"Feature"] = summary_train_continuous.loc[i,"Feature"]
        feature_summary.loc[i+numcat,"Type"] = "Continuous"
        feature_summary.loc[i+numcat,"#unique(Train)"] = "-"
        feature_summary.loc[i+numcat,"#unique(Test)"] = "-"
    
    #Print status if verbose true:
    if verbose:
        print ' Complete!'
        print 'Exporting Results as Excel File.... ',

    #Copy blank file from source directory:
    shutil.copyfile(os.path.join(module_location,"DataAnalysis.xlsx") ,os.path.join(os.getcwd(),"DataAnalysis.xlsx"))

    #Row index for export:
    row_ind = 1
    export_to_excel(OverallSummary, "raw_data",row_ind,1)
    export_to_excel(feature_summary, "raw_data",8,1)    

    row_ind += 10 + shape_train_continuous[1] + shape_train_categorical[1]
    export_to_excel(summary_train_categorical, "raw_data",row_ind,1)
    
    row_ind += shape_train_categorical[1] + 3
    export_to_excel(summary_test_categorical, "raw_data",row_ind,1)
    
    row_ind += shape_test_categorical[1] + 3
    export_to_excel(summary_combined_categorical, "raw_data",row_ind,1)
    
    row_ind += shape_combined_categorical[1] + 3
    export_to_excel(summary_train_continuous, "raw_data",row_ind,1)
    
    row_ind += shape_train_continuous[1] + 3
    export_to_excel(summary_test_continuous, "raw_data",row_ind,1)
    
    row_ind += shape_test_continuous[1] + 3
    export_to_excel(summary_combined_continuous, "raw_data",row_ind,1)
    
    row_ind += shape_combined_continuous[1] + 3
    export_to_excel(RelationalMatrix_train, "raw_data",row_ind,1)
    
    row_ind += shape_train_continuous[1] + shape_train_categorical[1] + 3
    export_to_excel(RelationalMatrix_test, "raw_data",row_ind,1)
    
    row_ind += shape_test_continuous[1] + shape_test_categorical[1] + 3
    export_to_excel(RelationalMatrix_combined, "raw_data",row_ind,1)

    #Print status if verbose true:
    if verbose:
        print ' Complete!' 
    