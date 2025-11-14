#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kangming
"""
#%%
''' Import packages '''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_predict
from catalysis_yang_project import read_synthesis_data, get_class_scores,loocv_train_predict
from featurizers import featurize_precursors, plot_cm,stoichiometry_featurizer
from sklearn.model_selection import KFold, StratifiedKFold



#%%
''' Read data '''
df = read_synthesis_data(
    filename='data/Sol-gel synthesis-Re activity.csv',
    skiprows=1,
    )

solvant = (df['C2H6O amount (mL)']+df['H2O amount (mL)'])
df['C2H6O amount (mL)'] = df['C2H6O amount (mL)'] / solvant
df['H2O amount (mL)'] = df['H2O amount (mL)'] / solvant


target = 'Synthesis'

df,ion_features, cation_features, anion_features = featurize_precursors(df)


#%%
stoichiometry_features = stoichiometry_featurizer(df['Formula'])
stoichiometry_features.index = df.index


# %%
y = df[target]
X = pd.concat([
    ion_features,
    cation_features, 
    df['H2O amount (mL)'],
    # anion_features, 
    # df['concentrations']
    ], axis=1)

#%%
cols2drop = [
    i for i in X.columns 
        if 'maximum' in i 
        or 'minimum' in i 
        or 'std_dev' in i
        or 'possible' in i
        or ('avg' in i and 'affinity' not in i)
        or 'cation' in i
        or ('anion' in i and 'affinity' not in i)
    ]
cols2drop = [i for i in cols2drop if 'std_dev EN' not in i] + ['ion range oxidation state']
X = X.drop(columns=cols2drop)

# X = stoichiometry_features

#%%
from elementembeddings.composition import composition_featuriser
# from elementembeddings.core import Embedding

# # # Embedding.load_data()
for embedding in ['megnet16']: # 'megnet', 'mat2vec''matscholar'

    # df_featurised = composition_featuriser(
    #     df.reset_index()['Formula'], 
    #     # df.reset_index()['Precursor combined'].astype(str), 
    #     embedding=embedding, #mat2vec
    #     stats=["mean","variance","minpool",'maxpool'], # 'maxpool' seems to be important
    #     )
    # df_featurised.index = df.index
    # df_featurised = df_featurised.drop(columns=['formula'])
    # df_featurised.columns = [f'{embedding} {i}' for i in df_featurised.columns]
    # X = pd.concat([X, df_featurised], axis=1)


    df_featurised = composition_featuriser(
        df.reset_index()['Precursor combined'].astype(str), 
        embedding=embedding, #mat2vec
        stats=["mean","variance","minpool",'maxpool']
        )
    df_featurised.index = df.index
    df_featurised = df_featurised.drop(columns=['formula'])

    X = pd.concat([X, df_featurised], axis=1)




#%%
''' Define the model '''

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ratio_0 = y.value_counts()[0]/y.value_counts().sum()
ratio_1 = y.value_counts()[1]/y.value_counts().sum()
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

''' Random Forest Classifier '''

rfc = RandomForestClassifier(n_estimators=100, 
                             random_state=0, 
                             n_jobs=-1,
                             max_features=0.3, 
                             class_weight= 'balanced',#{0:ratio_1*1.5,1:ratio_0}, #'balanced', #{0:7/9,1:2/9}
                             bootstrap=False,
                             )
clf['rfc'] = Pipeline([
    ('scaler', StandardScaler()),
    ('model', rfc),
    ])


#%%
print(X.shape)
# y_pred,y_pred_proba = loocv_train_predict(clf['xgb'], X, y,pbar=False)

# cross validation y_pred_proba using sklearn
# kf = StratifiedKFold(n_splits=X.shape[0], random_state=0, shuffle=True)
kf = KFold(n_splits=20, random_state=0, shuffle=True)
y_pred_proba = cross_val_predict(clf['xgb'], X, y, cv=kf, method='predict_proba')
y_pred_proba = pd.DataFrame(y_pred_proba, index=y.index)

#%%
threshold = 0.5 # higher threshold means higher tolerance for false negatives
y_pred = y_pred_proba[1].apply(lambda x: 1 if x>threshold else 0)
class_scores = get_class_scores(y,y_pred,y_pred_proba)

#%%


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
        
    # set the ticks. Modify the labels to show 'gelable' and 'non-gelable' instead of '1' and '0'
    plt.yticks([0,1], labels)
    plt.xticks([0,1], labels)
    # set the ticks. Modify the labels to show 'gelable' and 'non-gelable' instead of '1' and '0'
    plt.yticks([0,1], ['Non-gelable','Gelable'],fontsize=fontsize-2)
    plt.xticks([0,1], ['Non-gelable','Gelable'],fontsize=fontsize-2)
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
fig.savefig(f'confusion_matrix_synthesis.png',dpi=300,bbox_inches='tight')


#%%
# plot ROC curve

fontsize =10

fpr, tpr, thresholds = roc_curve(y, y_pred_proba[1])
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(3.5,3.5))
lw = 2
plt.plot(fpr, tpr, 'o-', color='darkorange',
         markerfacecolor='w',
        lw=lw, label='ROC curve')
# plt.text(0.15, 0.65, 'AUC = %0.3f' % roc_auc, fontsize=fontsize,color='darkorange')
plt.text(0.025, 0.95, 'AUC = %0.3f' % roc_auc, fontsize=fontsize,color='darkorange')

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--',label='Random guess')

# # mark the classification threshold on the curve
# for threshold in [0.5,0.7]:
#     close_default_clf = np.argmin(np.abs(thresholds - threshold))
#     plt.plot(fpr[close_default_clf], tpr[close_default_clf], '^', markersize=6,
#             # label="Threshold = "+str(threshold), 
#             fillstyle="none", c='k', mew=1.5)
#     # add the threshold as text
#     plt.text(fpr[close_default_clf], 
#              tpr[close_default_clf]-0.06, 
#              threshold, fontsize=10,color='k')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False gelable rate')
plt.ylabel('True gelable rate')
# plt.title('Receiver operating characteristic curve')
plt.legend(loc="lower right")
plt.show()
plt.savefig(f'roc_synthesis.png',dpi=300,bbox_inches='tight')




#%%
y = df[target]
X_shap = pd.concat([
    ion_features,
    cation_features, 
    df['H2O amount (mL)'],
    # anion_features, 
    # df['concentrations']
    ], axis=1)

model = clf['xgb']

cols2drop = [
    i for i in X_shap.columns 
        if 'maximum' in i 
        or 'minimum' in i 
        or 'std_dev' in i
        or 'possible' in i
        # or 'avg' in i
        # or 'cation' in i
    ]
cols2drop = [i for i in cols2drop if 'std_dev EN' not in i] + ['ion avg ionic char'] + ['ion range oxidation state']
X_shap = X_shap.drop(columns=cols2drop)
# rename the columns using dict
dict_rename = {
    'H2O amount (mL)': 'H$_2$O Fraction',
    'ion range oxidation state': 'Oxid. Range',
    'cation range oxidation state': 'Cation Oxid. Range',
    'ion std_dev EN difference': r'StDev $\Delta \chi$',
    'ion range EN difference': r'$\Delta \chi$ Range',
    'ion max ionic char': 'Max. Ionic Char.',
    'ion avg anion electron affinity': 'Avg. Anion EA',
}
X_shap = X_shap.rename(columns=dict_rename)

X_shap.columns


#%%
'''
Some checking:
originally I just want to check how accurate the model is,
'''
# kf = KFold(n_splits=5, random_state=1, shuffle=True)
kf = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
y_pred_proba = cross_val_predict(clf['xgb'], X_shap, y.loc[X_shap.index],
# y_pred_proba = cross_val_predict(clf['xgb'], X, y.loc[X.index],
                                #   cv=50, 
                                cv = kf,
                                method='predict_proba')

y_pred_proba = pd.DataFrame(y_pred_proba, index=y.index)

threshold = 0.5 # higher threshold means higher tolerance for false negatives
y_pred = y_pred_proba[1].apply(lambda x: 1 if x>threshold else 0)

class_scores = get_class_scores(y,y_pred,y_pred_proba)

#%%
regressor = clf['xgb'].named_steps['model']
regressor.fit(X_shap, y.loc[X_shap.index])

#%%
import shap 
shap_values = shap.TreeExplainer(regressor).shap_values(X_shap)
shap.summary_plot(
    shap_values, 
    X_shap,
    plot_type="dot",
    show=False,
                  )

# get the figure and axis
fig, ax = plt.gcf(), plt.gca()
# set the size of the figure
fig.set_size_inches(6, 4.5)
ax.set_xlabel('SHAP Value')
plt.tight_layout()
plt.savefig('gelation_shap_plot.png', dpi=300)


# %%
