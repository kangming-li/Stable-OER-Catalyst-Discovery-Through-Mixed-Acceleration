#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kangming
"""


#%%
''' Import packages '''
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import cross_val_predict
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold

from catalysis_yang_project import read_performance_data,get_class_scores
from featurizers import featurize_precursors
tqdm.pandas()

def get_scores(y_true, y_pred):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from scipy.stats import pearsonr
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    if len(y_true) == 1:
        r2 = None
        r_pearson = None
        print(f'RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: /, MAPE: {mape:.2f}%, Pearson: /')
    else:
        r2 = r2_score(y_true, y_pred)
        # get pearson correlation coefficient
        r_pearson = pearsonr(y_true, y_pred)[0]
        print(f'RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}, MAPE: {mape:.2f}%, Pearson: {r_pearson:.2f}')
    return rmse, mae, mape, r2, r_pearson



#%%
''' Read data '''
OP, stability = 'OP (10mA/cm2)', 'ICP-2h (%)' #'ICP-2h (%)', 'ICP-2h (%) XRF'

target = OP

df = read_performance_data(
    filename='data/metal dissolution-20250506.csv',
    target=target,
    additional_cols=[f'{i} element' for i in [1,2,3]]+['C2H6O amount (mL)','H2O amount (mL)'],
    performance_classification=True,
    )
#%%
solvant = (df['C2H6O amount (mL)']+df['H2O amount (mL)'])
df['C2H6O amount (mL)'] = df['C2H6O amount (mL)'] / solvant
df['H2O amount (mL)'] = df['H2O amount (mL)'] / solvant


#%%
'''
Create a new dataframe where the values in the columns 'amount of precursor i (mmol)' 
are replaced by the values in the columns 'i element'. This is to replace the nominal 
concentration by the XRF measured concentration.
'''
df_xrf = df.copy(deep=True)

for i in range(1,4):
    df_xrf.loc[
        df_xrf['1 element'].notna(),
        'amount of precursor '+str(i)+' (mmol)'
        ] = df_xrf.loc[
            df_xrf['1 element'].notna(),
            f'{i} element'
            ] 

#%%
# reset the index
df = df.reset_index()
df_xrf = df_xrf.reset_index()

cols_xrd = []

#%% Add XRD features

# read XRD features, including structural features from matminer
df_xrd = pd.read_json('data/structural_features.json')

# drop rows that have duplicated index
df_xrd = df_xrd[~df_xrd.index.duplicated(keep='first')]

# replace empty string with np.nan
df_xrd = df_xrd.replace({'': np.nan})

# normalize the relative intensity
total_area = df_xrd[[f'area {i}' for i in range(1,4)]].sum(axis=1)
for i in range(1,4):
    df_xrd[f'area {i}'] = df_xrd[f'area {i}']/total_area



#%%
# drop relative_intensity 
df_xrd = df_xrd.drop(columns=[i for i in df_xrd.columns if 'relative_intensity' in i])

index = df_xrd.index.intersection(df['XRD file number'].dropna())
df_xrd = df_xrd.loc[index]

df = df.merge(df_xrd, left_on='XRD file number', right_index=True, how='left')


#%%
df_xrf, ion_features_xrf, cation_features_xrf, anion_features_xrf = featurize_precursors(df_xrf)
df,ion_features, cation_features, anion_features = featurize_precursors(df)

other_features = [
    'amount (mL)', 
    'Anneal temp', 'Annealing time (h)',
    'Wash with Actone and H2O',
    'C2H6O amount (mL)','H2O amount (mL)',
] + cols_xrd 
df_other_features = df[other_features]

#%% Set X and y

X = pd.concat([
    ion_features,
    cation_features, 
    df_other_features,
    # cation_features_xrf,
    # ion_features_xrf,
    # anion_features_xrf,
    ], axis=1).astype(float)

# removed_features = [
#        'amount (mL)',
#        'ion compound possible',
#        'C2H6O amount (mL)',
#        'Wash with Actone and H2O','cation range oxidation state',
#        ] + (
#            [i for i in X.columns if 'maximum' in i] 
#            + [i for i in X.columns if 'minimum' in i]
#            + [i for i in X.columns if 'std_dev' in i]
#            + [i for i in X.columns if 'avg' in i]
#        ) 
# # remove 'ion std_dev EN difference' from removed_features
# removed_features = [i for i in removed_features if 'ion std_dev EN difference' not in i]
# removed_features.append('ion range EN difference')
# X = X.drop(columns=removed_features)


y = df[target].astype(float)
index = X.dropna().index
X = X.loc[index]
y = y.loc[index]


y_threshold = 220
y = y < y_threshold

ratio_1 = y.sum()/y.count()
ratio_0 = 1 - ratio_1

#%%
''' Define the model '''

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Lasso
import xgboost as xgb


def get_model(modelname,random_state):

    clf = {}

    ''' XGBoost Classifier '''

    rfc = xgb.XGBClassifier(
        n_jobs=-1, random_state=0,
        n_estimators=200, learning_rate=0.2,
        reg_lambda=0.5,reg_alpha=0.5,
        colsample_bytree=0.5,colsample_bylevel=0.5,
        num_parallel_tree=5,
        scale_pos_weight=ratio_0/ratio_1,
                                )
    # include a standard scaler in the pipeline
    clf['xgb'] = Pipeline([
        ('scaler', StandardScaler()),
        ('model', rfc),
        ])

    return clf[modelname]


#%% get the RMSE, MAE, R2, and pearson correlation coefficient
modelname = 'xgb'

random_state = 1
model = get_model(modelname=modelname,random_state=random_state)
kf = KFold(n_splits=10, random_state=random_state, shuffle=True)
y_pred_proba = cross_val_predict(model, X, y, cv=kf, method='predict_proba')
y_pred_proba = pd.DataFrame(y_pred_proba, index=y.index)

threshold = 0.5 # higher threshold means higher tolerance for false negatives
y_pred = y_pred_proba[1].apply(lambda x: 1 if x>threshold else 0)
class_scores = get_class_scores(y,y_pred,y_pred_proba)



def plot_cm(y, y_pred, labels=[0,1], fontsize=20,text=None):
    from sklearn.metrics import confusion_matrix
    # use plt to plot the confusion matrix, and add the values on the plot
    cm = confusion_matrix(y, y_pred,labels=labels)
    cm_normalized = confusion_matrix(y, y_pred,labels=labels, normalize='true')
    
    # use a light color map
    plt.imshow(cm, cmap=plt.cm.Blues)
    fig = plt.gcf()
    # change figsize
    fig.set_size_inches(5,5)

    ax = plt.gca()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if (i,j) == (1,1):
                color = 'white'
            else:
                color = 'black'

            plt.text(j, i, cm[i,j], ha='center', va='center',fontsize=fontsize,color=color)
            # plt.text(j, i+0.1, 
            #          f'({cm_normalized[i,j]*100:.0f}%)', 
            #          ha='center', va='center',
            #          fontsize=fontsize,color=color)

    # add precision and recall
    precision = cm[1,1]/cm[:,1].sum()
    recall = cm[1,1]/cm[1].sum()
    plt.text(1,1+0.2,f'Precision: {precision*100:.0f}%',
             ha='center', va='center',fontsize=fontsize,color=color)
    plt.text(1,1+0.31,f'Recall: {recall*100:.0f}%',
             ha='center', va='center',fontsize=fontsize,color=color)
        
    # set the ticks. Modify the labels to show 'Low OP' and 'High OP' instead of '1' and '0'
    plt.yticks([0,1], labels)
    plt.xticks([0,1], labels)
    # set the ticks. Modify the labels to show 'Low OP' and 'High OP' instead of '1' and '0'
    plt.yticks([0,1], ['High OP','Low OP'],fontsize=fontsize-2)
    plt.xticks([0,1], ['High OP','Low OP'],fontsize=fontsize-2)
    # rotate the yticks and center them
    plt.yticks(rotation=90, va='center')
    # set the fontsize of the labels in the axes
    plt.xlabel('Predicted label', fontsize=fontsize)
    plt.ylabel('True label', fontsize=fontsize)
    # plt.title('Confusion matrix', fontsize=fontsize)
    # set xtick in the upper x axis
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    if text:
        ax.text(0.1, -0.1, text, fontsize=fontsize-2,transform=ax.transAxes)

    plt.tight_layout()
    plt.show()
    # return the figure
    return fig, ax


fig,ax = plot_cm(y, y_pred, fontsize=15,labels=[0,1])


#%%



#%%
