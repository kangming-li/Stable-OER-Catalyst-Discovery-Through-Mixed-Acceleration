#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kangming
"""

#%%
''' Import packages '''
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import LeaveOneOut
import re
from sklearn.metrics import (roc_curve, auc,
                            accuracy_score, balanced_accuracy_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef, 
                             average_precision_score, roc_auc_score)
import matplotlib.pyplot as plt

#%%
''' Read synthesis data '''

def read_synthesis_data(
        filename='Sol-gel synthesis.csv',
        max_nelements=None,
        skiprows=1,
        ):
    # skip the first row of the csv file
    df = pd.read_csv(filename,skiprows=skiprows).set_index('Synthesis number')

    # Drop rows whose values in 'Synthesis' are Nan
    df = df.dropna(subset=['Synthesis'])

    cols2keep = [
    'Formula', 
    'Precursor 1', 'amount of precursor 1 (mmol)', 
    'Precursor 2', 'amount of precursor 2 (mmol)', 
    'Precursor 3', 'amount of precursor 3 (mmol)', 
    'Precursor 4', 'amount of precursor 4 (mmol)', 
    # 'Solvent', 
    # 'Solvent amount (mL)',
    # 'Gelation agent', 
    'amount (mL)', 
    'Synthesis',
    ] + ['C2H6O amount (mL)','H2O amount (mL)']
    df = df[cols2keep]


    solvant = (df['C2H6O amount (mL)']+df['H2O amount (mL)'])
    df['C2H6O amount (mL)'] = df['C2H6O amount (mL)'] / solvant
    df['H2O amount (mL)'] = df['H2O amount (mL)'] / solvant


    if max_nelements == 2:
        # consider only the binary synthesis
        # drop rows where the values in 'Precursor 3' are Nan
        df = df[df[['Precursor 3']].isna().any(axis=1)]


    # Preprocessing the data

    # substitue the values '[(CH3)2CHO]2Ti(C5H7O2)2' as '(CH3)4C2H2O2Ti(C5H7O2)2'
    # This is because the use of '[]' in the formula is not recognized by the Composition class
    df = df.replace('[(CH3)2CHO]2Ti(C5H7O2)2', '(CH3)4C2H2O2Ti(C5H7O2)2')

    # changes all values "H" in column 'Synthesis' to "C"
    df['Synthesis'] = df['Synthesis'].replace('H', 'C')
    # replace all values "C" (including preceding and trailing spaces) 
    # in column 'Synthesis' to 1
    df['Synthesis'] = df['Synthesis'].replace(r'\s*C\s*', 1, regex=True)
    # replace all values "N" (including preceding and trailing spaces)
    # in column 'Synthesis' to 0
    df['Synthesis'] = df['Synthesis'].replace(r'\s*N\s*', 0, regex=True)
    # # sort the dataframe by the column 'Synthesis'
    # df = df.sort_values(by=['Synthesis'], ascending=False)

    # drop repeated rows
    subset = [i for i in df.columns if i not in ['Synthesis']]
    df = df.drop_duplicates(subset=subset, keep='first')

    # find duplicated rows and sort by the column 'Formula'
    df[df.duplicated(subset='Formula', keep=False)].sort_values(by=['Formula'])

    return df

'''
Read OER performance and stability data
'''
def read_performance_data(
        filename='Sol-gel synthesis-Re activity.csv',
        skiprows=1,
        target = None,
        performance_classification = False,
        OP_max = 350,
        additional_cols = [],
        ):
    # skip the first row of the csv file
    df = pd.read_csv(filename,skiprows=skiprows).set_index('Synthesis number')


    target1 = 'OP (10mA/cm2)'

    if target == target1:
        additional_cols += [target]
    else:
        additional_cols += [target,'OP (10mA/cm2)']
        

    df = df.replace('[(CH3)2CHO]2Ti(C5H7O2)2', '(CH3)4C2H2O2Ti(C5H7O2)2')
    df = df.replace('#REF!', np.nan).replace('#VALUE!', np.nan)

    cols2keep = [
        'Formula', 
        'Precursor 1', 'amount of precursor 1 (mmol)', 
        'Precursor 2', 'amount of precursor 2 (mmol)', 
        'Precursor 3', 'amount of precursor 3 (mmol)', 
        'Precursor 4', 'amount of precursor 4 (mmol)', 
        'XRD file number',
        'amount (mL)', 'Anneal temp', 'Annealing time (h)',
        'Wash with Actone and H2O',
        ] +  additional_cols


    df = df[cols2keep]

    # Preprocessing

    # Drop rows whose values in 'OP (10mA/cm2)' are empty
    # This processing is for both performance and stability
    df = df.dropna(subset=['OP (10mA/cm2)'])

    

    # # find duplicated rows and sort by the column 'Formula'
    # df[df.duplicated(subset='Formula', keep=False)].sort_values(by=['Formula'])
        

    # df = df.sort_values(by=[target])

    if target == 'OP (10mA/cm2)':
        # For OP measurement:
        # check whether the values in the column are numeric
        # and replace the non-numeric values with 500
        df.loc[~df[target].apply(lambda x: x.isnumeric()),target] = 500
        df[target] = df[target].astype('float')

        # remove all the rows where the values in the column OP are greater than 350
        OP_max = OP_max #400
        OP_min = 100 # why do we want to remove this? there is one data
   
        if performance_classification:
            return df
        else:
            df = df[(df[target] <= OP_max) & (df[target] >= OP_min)] 
            print('Returning OP data between {} and {}'.format(OP_min,OP_max))
            return df 
    else:
        df = df.dropna(subset=[target])
        return df




#%%
''' Leave one out cross validation '''
def loocv_train_predict(model, X, y,regression=False,pbar=True):
    # Do LOOCV on (X,y), get the CV predicted values, print the mean accuracy, and plot the confusion matrix
    loo = LeaveOneOut()
    y_pred = pd.Series(index=y.index)

    if not regression:
        # create a 2-column dataframe to store the predicted probabilities 
        y_pred_proba = pd.DataFrame(index=y.index, columns=[0,1])

    if pbar:
        iterator = tqdm(loo.split(X))
    else:
        iterator = loo.split(X)

    # show the progress bar
    for train_index, test_index in iterator:
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train.to_numpy(), y_train)
        y_pred.iloc[test_index] = model.predict(X_test.to_numpy())[0]
        # print(test_index, y_pred.iloc[test_index])

        if not regression:
            y_pred_proba.iloc[test_index] = model.predict_proba(X_test)[0] 

    if regression:
        return y_pred
    else:
        return y_pred, y_pred_proba



#%%
def get_class_scores(y, y_pred, y_pred_proba, print_scores=True):
    '''
    Compare the performance with a classifier that always predicts 1
    round the scores to 3 decimal places
    '''
    # Calculate scores
    mean_accuracy = round(accuracy_score(y, y_pred), 3)
    balanced_accuracy = round(balanced_accuracy_score(y, y_pred), 3)
    precision = round(precision_score(y, y_pred), 3)
    recall = round(recall_score(y, y_pred), 3)
    specificity = round(recall_score(y, y_pred, pos_label=0), 3)
    g_mean = round(np.sqrt(specificity * recall), 3)
    f1 = round(f1_score(y, y_pred), 3)
    f1_negative = round(f1_score(y, y_pred, pos_label=0), 3)
    mcc = round(matthews_corrcoef(y, y_pred), 3)
    avg_precision_score = round(average_precision_score(y, y_pred), 3)
    auc = round(roc_auc_score(y, y_pred_proba.loc[:, 1]), 3)

    scores = {
        'Mean accuracy': mean_accuracy,
        'Balanced accuracy': balanced_accuracy,
        'Precision': precision,
        'Recall (sensitivity)': recall,
        'Specificity': specificity,
        'G-mean': g_mean,
        'F1 score': f1,
        'F1 score (negative)': f1_negative,
        'MCC score': mcc,
        'Average precision score': avg_precision_score,
        'AUC score': auc,
    }

    if print_scores:
        for score_name, score_value in scores.items():
            print(f'{score_name}: {score_value}')


    return scores

#%%
def plot_ROC_AUC(y,y_pred_proba,thresholds2mark=[0.5,0.7]):

    # plot ROC curve

    fontsize =10

    fpr, tpr, thresholds = roc_curve(y, y_pred_proba[1])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(3.75,3.75))
    lw = 2
    plt.plot(fpr, tpr, 'o-', color='darkorange',
            lw=lw, label='ROC curve')
    # plt.text(0.15, 0.65, 'AUC = %0.3f' % roc_auc, fontsize=fontsize,color='darkorange')
    plt.text(0.025, 0.95, 'AUC = %0.3f' % roc_auc, fontsize=fontsize,color='darkorange')

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--',label='Random guess')

    # mark the classification threshold on the curve
    for threshold in thresholds2mark:
        close_default_clf = np.argmin(np.abs(thresholds - threshold))
        plt.plot(fpr[close_default_clf], tpr[close_default_clf], '^', markersize=6,
                # label="Threshold = "+str(threshold), 
                fillstyle="none", c='k', mew=1.5)
        # add the threshold as text
        plt.text(fpr[close_default_clf], 
                tpr[close_default_clf]-0.06, 
                threshold, fontsize=10,color='k')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False synthesizable rate')
    plt.ylabel('True synthesizable rate')
    # plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    plt.show()




#%% Pourbaix diagram


#%% Processing



