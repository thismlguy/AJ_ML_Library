###############################################################################
##### SUMMARY (HELP)
###############################################################################

"""
Function1: bivariate(col1, col2, col1Type='infer', col2Type='infer')
    used to generate plots corresponding to a pair of columns
    Arguments: 
        a. col1: Series object - the first column to be analyzed 
        b. col2: Series object - the second column to be analyzed

    Output: 
        a. Chart corresponding to the data

Function2: univariate(col, colType='infer', transformation=None, param={},check_duplicate_categories=False,return_mod=False)
    used to analuze 1 variable at a time. Note: This doesn't change the original variable 
    Arguments: 
        a. col: Series object - the column to be analyzed 
        b. colType: string constant - 2 options: 'categorical' or 'continuous' 
        c. transformation: string constant - depending on colType, it can be None or one of following:
            i. 'continuous' : log, square, square root, cube, cube root, combine
            ii. 'categorical' : combine
        d. param: the parameters required for the transformation type selected
            i. if 'continuous' & 'combine': Pass a list of intermediate cut-points. Min and Max will automatically added
            ii. if 'categorical' & 'combine' : Pass a dictionary in format - {'new_category':[list of categories to combines]}
        e. check_duplicate_categories: Applicable only for categorical varaible. Checks if the categories are different only by upper or lower case and resolves the same. Eg - 'High' and 'high' will be resolved to 'high'
        f. return_mod: if True returns the modified variable which can be used to create a new variable or replace the old one
    
    Output: 
        a. Chart corresponding to the data
        b. The modified variable if return_mod is True

Funtion3: def imputation(data, col, method, param={}, colType='infer')
    Function to perform imputation. 
    Arguments:
        a. data: the full data frame whose column is to be modified
        b. col: name of the column to be modified
        c. method: differs depending on type of variable
            for continuous:
                1. mean - impute by mean
                2. median - impute by median
            for categorical:
                1. mode - impute by mode
                2. category - fixed value impute
        d. param: dictionary of the additional parameters to be used:
            'groupby':colname - the colum by which the method is to be grouped for imputation
        c. colType: type of the column. if it is 'infer', it will be selected based on datatype
            This is provided to account for categorical variables like 'H-M-L' which are coded as '0-1-2' and will be treated numerical by default
   
"""

###############################################################################
##### IMPORT STANDARD MODULES
###############################################################################

# %matplotlib inline    #only valid for iPython notebook
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import feature_selection
from scipy.stats import chisquare, linregress, mode

###############################################################################
##### DEFINE PLOTTING FUNCTIONS
###############################################################################

#Function to plot a bar chart
#Note: data should be a dataframe!
def bar_chart(plt, data, Xlabel=None, Ylabel=None, Title=None, Legend=True, Width=0.8, Align="center"):

    tableShape = data.shape
    colormap = plt.cm.get_cmap('Set1')
    
    ind = np.arange(tableShape[1])    # the x locations for the groups
    width = 0.4       # the width of the bars: can also be len(x) sequence

    # plt.figure()
    bottom = [0] * tableShape[1]
    
    c=0 #for color index
    multiplier = 1./tableShape[0]
    p = [] #for storing plots
    for i,row in data.iterrows():
        p.append(plt.bar(left=ind, height=row, width=Width, bottom=bottom, color=colormap(c),align=Align))
        c+=multiplier
        bottom = row
    
    if Ylabel:
        plt.ylabel(Ylabel)
    
    if Xlabel:
        plt.xlabel(Xlabel)
    
    if Title:
        plt.title(Title)
    
    plt.xticks(ind , (data.columns), rotation=30, ha="center")

    if Legend:
        plt.legend(p, data.index, loc="best", bbox_to_anchor=(1.7,-0.3))

#Function to plot a scatter chart
def scatter_chart(plt, col1, col2, Title="Scatter Plot"):
    color = ['r']
    results = linregress(col1,col2)
    print results
    plt.scatter(col1,col2)
    plt.plot(col1, col1*results[0] + results[1])
    plt.ylabel(col2.name)
    plt.xlabel(col1.name)
    plt.title(Title)
    # plt.set_label('R-squared = %f'%results[2])

#Function to plot boxplot
def boxplot_chart(plt, col1, col2=None, Title="BoxPlot"):
    #col1: numeric
    #col2: categorical

    if col2 is not None:
        group = {}
        for i in range(len(col2)):
            if col2.loc[i] in group:
                group[col2.loc[i]].append(col1.loc[i])
            else:
                group[col2.loc[i]] = [col1.loc[i]]
        
        # plt.figure()
        plt.boxplot([num for val, num in group.items()], showmeans=True )  
        plt.xlabel(col2.name)
    else:
        plt.boxplot(col1, showmeans=True )

    plt.ylabel(col1.name)
    plt.title(Title)
    # plt.set_label('R-squared = %f'%results[2])

#Function to plot histogram.
def histogram_chart(plt, col, Ylabel="Frequency", Xlabel=None, Title="Histogram"):
    col.dropna(inplace=True)
    
    plt.hist(col)
    
    if Ylabel:
        plt.ylabel(Ylabel)
    
    if Xlabel:
        plt.xlabel(Xlabel)
    
    plt.title(Title)        

###############################################################################
##### DEFINE SUPPORT FUNCTIONS
###############################################################################

#Plotting for case of both categorical variables
def chart_both_categorical(col1, col2):
    contingencyTable = pd.crosstab(index = col2, columns=col1)
    tableShape = contingencyTable.shape

    print contingencyTable
    if tableShape[0] > 10 or tableShape[1] > 10:
        print "Error: Too many categories. The unique categories should be less than 10 in each variable."
        return  

    plt.subplot(121)
    bar_chart(plt,contingencyTable, Xlabel=col1.name, Ylabel=col2.name, Title="Bar Chart (Absolute)")
    
    contingencyTable = contingencyTable.apply(lambda x: x/float(x.sum()))
    plt.subplot(122)
    bar_chart(plt,contingencyTable, Xlabel=col1.name, Ylabel=col2.name, Title="Bar Chart (Percentage)", Legend=False)

    plt.tight_layout()
    plt.show(block=False)

#Plot for case of both continuous variables.
def chart_both_continuous(col1, col2):
    #Col1: x-axis
    #Col2: y-axis

    plt.subplot(111)
    scatter_chart(plt,col1,col2)
    plt.show(block=False)

#Plot for case of combination of categorical and continuous
def chart_combo(col1, col2):
    #Col1: categorical
    #Col2: numeric
    plt.subplot(111)
    boxplot_chart(plt,col2, col1)
    plt.show(block=False)

#function to return data type (continuous/categorical):
def check_datatype(col):
    if col.dtypes == 'object':
        return 'categorical'
    else:
        return 'continuous'

###############################################################################
##### DEFINE FEATURE ENGINEERING FUNCTION
###############################################################################

#Comparison of two variables
def bivariate(col1, col2, col1Type='infer', col2Type='infer'):
    
    #If the datatypes not specified, determined them:
    if col1Type == 'infer':
        col1Type = check_datatype(col1)

    if col2Type == 'infer':
        col2Type = check_datatype(col2)    

    #Run the respective function depending on the type of each varaible:
    if col1Type == col2Type:
        if col1Type == "categorical":
            chart_both_categorical(col1, col2)
        else:
            chart_both_continuous(col1, col2)
    else:  
        if col1Type == "categorical":
            chart_combo(col1, col2)
        else:
            chart_combo(col2, col1)

#Univariate analysis:
def univariate(col, colType='infer', transformation=None, param={},check_duplicate_categories=False,return_mod=False):

    #Get the column type if it is infer
    if colType=='infer':
        colType = check_datatype(col)

    #Case of contiuous variable:
    if colType == "continuous":
        #Case of transformation as combine:
        if transformation == "combine":
            #check if cuts provided, return error if not
            if 'cuts' not in param:
                print "Parameter Missing! Atleast a 'cuts' parameter required which is a list of intermediate cut-points. Min and Max will automatically added"
                return
            
            #define break point:
            break_points = [col.min()] + param['cuts'] + [col.max()]

            #define labels for the binned data:
            if 'labels' in param:
                labels = param['labels']
            else:
                labels = ["%g - %g"%(break_points[i],break_points[i+1]) for i in range(len(break_points)-1)]

            #make a new Series object with the binned data 
            coltr = pd.cut(col,bins=break_points,right=True,labels=labels,include_lowest=True)

            #Get the frequency of each bin
            bar_data = pd.value_counts(coltr, sort=False)

            #Plot the bar-chart
            plt.subplot(111)
            bar_chart(plt, pd.DataFrame(bar_data.reshape(1,len(bar_data)), columns=bar_data.index), Xlabel=col.name, Title = "Binned Data", Legend=False)
            plt.show(block=False)
            
        else:
            #Check for mathematical transformations and apply them
            if transformation == "log":
                if min(col)<1:
                    coltr = np.log(col+1)
                else:
                    coltr = np.log(col)
            elif transformation == "square":
                coltr = np.power(col,2)
            elif transformation == "cube":
                coltr = np.power(col,3)
            elif transformation == "square root":
                coltr = np.power(col,0.5)
            elif transformation == "cube root":
                coltr = np.power(col,0.33)
            else:
                coltr = pd.Series(col,copy=True)
            
            #Determine skew and curtosis:
            skew = coltr.skew()
            kurt = coltr.kurt()
            title = "Skew = %g | Kurt = %g"%(np.round(skew,2),np.round(kurt,2))

            plt.subplot(121)
            histogram_chart(plt,coltr, Xlabel=col.name)

            plt.subplot(122)
            boxplot_chart(plt, coltr)
            plt.show(block=False)

    #Case of categorical variable:
    elif colType == "categorical":

        #Get the freq coutns of unique values:
        colUnq = pd.value_counts(col)
        #Initialize the transformed column to be the original column
        coltr = pd.Series(col, copy=True)

        #Case where different categories are to be combined
        if transformation == "combine":
            #Check if the combination dictionary is provided. Return error if not.
            if param is None:
                print "Parameter Missing! Pass a dictionary in format - {'new_category':[list of categories to combines]}"
                return

            #Use replace function to recode the values:
            for key,value in param.items():
                coltr.replace(value,[key]*len(value),inplace=True)
        
        #Get the new variable's freq counts:
        colUnq = pd.value_counts(coltr)

        #Perform check on duplicate classes:
        if check_duplicate_categories:
            #Check if some values are present which are duplicates except for difference in case:
            colUnqLower = colUnq.index.str.lower()
            colUnqLowerUnq = pd.value_counts(colUnqLower)

            duplicates = colUnqLowerUnq[colUnqLowerUnq>1]

            if not duplicates.empty:
                for s in duplicates.index:
                    for val in colUnq.index[colUnqLower == s]:
                        coltr.replace(val,s,inplace=True)
                bar_data = pd.value_counts(coltr)
                plt.subplot(122)
                bar_chart(plt, pd.DataFrame(bar_data.reshape(1,len(bar_data)), columns=bar_data.index), Xlabel=col.name, Ylabel=None, Title='Duplicates Combined', Legend=False)
                plt.show(block=False)
                plt.subplot(121)
        
        #Get the freq of modified column.
        bar_data = pd.value_counts(coltr)
        bar_chart(plt, pd.DataFrame(bar_data.reshape(1,len(bar_data)), columns=bar_data.index), Xlabel=col.name, Ylabel=None, Title='Original Data', Legend=False)
        plt.show(block=False)
        # print "Dulicate entries with different cases exist"
        # print pd.value_counts(coltr)

    #Return a value if needed
    if return_mod:
            return coltr

#Function to perform imputation. Arguments:
    # data: the full data frame whose column is to be modified
    # col: name of the column to be modified
    # colType: type of the column. if it is 'infer', it will be selected based on datatype
def imputation(data, col, method, param={}, colType='infer'):
    
    #Set the type of column:
    if colType=='infer':
        colType = check_datatype(data[col])

    #Initialize imputed column as the original column
    colimpt = pd.Series(data[col], copy=True)

    #Define a function to impute by groups if such selected
    def fill_grps(impute_grps):
        for i, row in data.iterrows():
            if pd.isnull(colimpt.loc[i]):
                x = tuple([ data.loc[i,x] for x in param['groupby']])
#                 print impute_grps
                colimpt.loc[i] = impute_grps.loc[x]
    
    #Case1: continuous column
    if colType == "continuous":
        #Impute by mean:
        if method == "mean":
            if param['groupby'] is not None:
                impute_grps = data.pivot_table(values=col, index=param['groupby'], aggfunc=np.mean, fill_value=data[col].mean())
                # print impute_grps
                fill_grps(impute_grps)
        
        #Impute by median:
        elif method == "median":
            if param['groupby'] is not None:
                impute_grps = data.pivot_table(values=col, index=param['groupby'], aggfunc=np.median, fill_value=data[col].mean())
                print impute_grps
                fill_grps(impute_grps)

        #Impute by model (Under development - To be ignored for now):
        # elif method == "model":
        #     miss_bool = data[col].isnull()
        #     train = data.loc[data[~miss_bool],[col]+param['predictors']]
        #     test = data.loc[data[miss_bool],[col]+param['predictors']]

    #Case2: Categorical variable:
    if colType == "categorical":

        #Impute by mode:
        if method == "mode":
            if param['groupby'] is not None:
                def cust_mode(x):
                    return mode(x).mode[0]
                impute_grps = data.pivot_table(values=col, index=param['groupby'], aggfunc=cust_mode, fill_value=cust_mode(data[col]))
#                 print impute_grps
                fill_grps(impute_grps)

        #Impute by fixed category
        elif method == "category":
            colimpt.fillna(param['category_name'], inplace=True)
        
        #Impute by model (Under development - To be ignored for now):
        # elif method == "model":
        #     miss_bool = data[col].isnull()
        #     train = data.loc[data[~miss_bool],[col]+param['predictors']]
        #     test = data.loc[data[miss_bool],[col]+param['predictors']]
            
    return colimpt 

