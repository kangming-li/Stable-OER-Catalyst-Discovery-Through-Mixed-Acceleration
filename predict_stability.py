#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#%%
''' Import packages '''
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import cross_val_predict
from matplotlib import pyplot as plt

from catalysis_yang_project import read_performance_data
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


use_xrd_features = 'matminer'  # 'matminer' # 'all' or 9 or 'onehot' or 'matminer'
use_OP = True

#%%
''' Read data '''
OP, stability = 'OP (10mA/cm2)', 'ICP-2h (%)' #'ICP-2h (%)', 'ICP-2h (%) XRF'

voltage = '2h voltage '
target = stability

df = read_performance_data(
    filename='data/Sol-gel synthesis-Re activity.csv',
   
    target=target,
    additional_cols=(
        [f'{i} element' for i in [1,2,3]]
        +[voltage]
        +['C2H6O amount (mL)','H2O amount (mL)']
        # +['ICP-2h (%) XRF']
    )
        ,
    OP_max=350,
    )

#%%

# df['ICP-2h (%) XRF'] = df['ICP-2h (%) XRF'].fillna(df['ICP-2h (%)'])
# df['ICP-2h (%)'] = df['ICP-2h (%) XRF']




#%%
solvant = (df['C2H6O amount (mL)']+df['H2O amount (mL)'])
df['C2H6O amount (mL)'] = df['C2H6O amount (mL)'] / solvant
df['H2O amount (mL)'] = df['H2O amount (mL)'] / solvant

if target == stability:
    # remove data whose values in voltage are > 1.448
    df = df[df[voltage] <= 1.448]

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

# Get the number of phases
phase_numbers = []
for index, row in df.iterrows():
    # check if Phase 3 is nan
    if row['Phase 3']==row['Phase 3']:
        phase_numbers.append(3)
    elif row['Phase 2']==row['Phase 2']:
        phase_numbers.append(2)
    elif row['Phase 1']==row['Phase 1']:
        phase_numbers.append(1)
    else:
        phase_numbers.append(0)
cols_xrd += ['phase_numbers']


df['phase_numbers'] = phase_numbers
if use_xrd_features == 'matminer':

    def get_area_weighted_mean(row,label):
        matminer_features = row['structural_features']
        values = [i[label] for i in matminer_features]
        # replace None by 0
        values = [i if i is not None else 0 for i in values]
        areas = [row[f'area {i}'] for i in range(1,4)]
        # replace np.nan by 0 
        areas = [i if i==i else 0 for i in areas]
        mean = 0
        for area, value in zip(areas,values):
            mean += area*value
        return mean

    arbitary_entry = df_xrd['structural_features'].iloc[0][0]
    matminer_feature_labels = list(arbitary_entry.keys())
    # drop 'vbm' from matminer_feature_labels
    # matminer_feature_labels = [i for i in matminer_feature_labels if i != 'vbm']

    matminer_feature_labels = [i for i in matminer_feature_labels if 'G_pbx' in i]

    matminer_cols = []
    for label in matminer_feature_labels:
        weighted_label = 'area weighted '+label
        df_xrd[weighted_label] = df_xrd.apply(
            lambda row: get_area_weighted_mean(row,label), axis=1
        )
        matminer_cols.append(weighted_label)

    df = df.merge(df_xrd[matminer_cols], left_on='XRD file number', right_index=True, how='left')
    cols_xrd += matminer_cols

elif use_xrd_features == 'none':
    cols_xrd = []

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


removed_features = [
    'area weighted G_pbx -1 0', 'area weighted G_pbx -1 1.5',
       'area weighted G_pbx 0 -1.5', 'area weighted G_pbx 0 0',
       'area weighted G_pbx 0 1.5', # This gets the correct physical interpretation and the performance is slightly worse
       'area weighted G_pbx 1 -1.5',
       'area weighted G_pbx 1 0', 
       'area weighted G_pbx 1 1.5',
    #    'area weighted G_pbx -1 -1.5', # This is used to get the best performance, but not used in SHAP
       'amount (mL)',

       'ion compound possible',
       'C2H6O amount (mL)','phase_numbers',
       'Wash with Actone and H2O','cation range oxidation state',
       ] + (
           [i for i in X.columns if 'maximum' in i] 
           + [i for i in X.columns if 'minimum' in i]
           + [i for i in X.columns if 'std_dev' in i]
           + [i for i in X.columns if 'avg' in i]
       ) 
# remove 'ion std_dev EN difference' from removed_features
removed_features = [i for i in removed_features if 'ion std_dev EN difference' not in i]
removed_features.append('ion range EN difference')
# X['diff G_pbx'] = X['area weighted G_pbx 0 1.5'] - X['area weighted G_pbx -1 -1.5']

X = X.drop(columns=removed_features)

y = df[target].astype(float)
index = X.dropna().index
X = X.loc[index]
y = y.loc[index]


#%%
'''
Add OP for the stability model
'''
if target == stability and use_OP:
    X = pd.concat([X,df.loc[index,OP].replace('N',np.nan).astype(float)],axis=1)

# X = X.drop(columns = ['area weighted G_pbx -1 -1.5', 'OP (10mA/cm2)'])



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

    n_estimators = 200
    num_parallel_tree = 5
    learning_rate = 0.2

    colsample_bytree = 0.3
    colsample_bylevel = 0.3

    reg_lambda = 0.5
    reg_alpha = 0.5

    # xgb regressor
    xgb_re = xgb.XGBRegressor(
        n_jobs=-1, random_state=random_state,
        n_estimators=n_estimators, learning_rate=learning_rate,
        reg_lambda=reg_lambda,reg_alpha=reg_alpha,
        colsample_bytree=colsample_bytree,colsample_bylevel=colsample_bylevel,
        num_parallel_tree=num_parallel_tree,
        objective = 'reg:squarederror',
    )


    if target == stability:
        # include a log transformer for the target values in the pipeline
        clf['xgb'] = Pipeline([
            ('scaler', StandardScaler()),
            ('model', TransformedTargetRegressor(
                regressor=xgb_re,
                func = np.log10,
                inverse_func = lambda x: 10**x,
                )),
            ])
    else:
        clf['xgb'] = Pipeline([
        ('scaler', StandardScaler()),
        ('model', xgb_re),
        ])



    ''' Random Forest regressor '''

    rfr = RandomForestRegressor(n_estimators=200,
                                random_state=0,
                                n_jobs=-1,
                                max_features=0.3,
                                bootstrap=False,
                                )

    if target == stability:
        # include a log transformer for the target values in the pipeline
        clf['rfr'] = Pipeline([
            ('scaler', StandardScaler()),
            ('model', TransformedTargetRegressor(
                regressor=rfr,
                # func=np.log,
                # inverse_func=np.exp,
                func = np.log10,
                inverse_func = lambda x: 10**x,
                )),
            ])
    else:
        clf['rfr'] = Pipeline([
            ('scaler', StandardScaler()),
            ('model', rfr),
            ])
        
    
    if target == stability:
        # include a log transformer for the target values in the pipeline
        clf['lasso'] = Pipeline([
            ('scaler', StandardScaler()),
            ('model', TransformedTargetRegressor(
                regressor=Lasso(alpha=0.1),
                func = np.log10,
                inverse_func = lambda x: 10**x,
                )),
            ])
    else:
        clf['lasso'] = Pipeline([
            ('scaler', StandardScaler()),
            ('model', Lasso(alpha=0.01)),
            ])


    return clf[modelname]


#%% get the RMSE, MAE, R2, and pearson correlation coefficient
from sklearn.model_selection import KFold
modelname = 'xgb'

n_repeat = 10

y_preds = []
maes = []
r2s = []
mapes = []
r_pearsons = []
models = []
for random_state_kf in range(n_repeat):
    # Create a KFold object
    kf = KFold(n_splits=20, random_state=random_state_kf, shuffle=True)
    model = get_model(modelname=modelname,random_state=random_state_kf)
    y_pred = cross_val_predict(model, X, y, cv=kf)
    y_pred = pd.Series(y_pred, index=y.index)

    # first log10 transform the target values
    _, _, _, r2, r_pearson = get_scores(
        np.log10(y), np.log10(y_pred)
        )
    rmse, mae, mape, _, _ = get_scores(y, y_pred)

    y_preds.append(y_pred)
    maes.append(mae)
    r2s.append(r2)
    mapes.append(mape)
    r_pearsons.append(r_pearson)
    models.append(model)

#%%    

fontsize = 12

y_pred = pd.Series(np.mean(y_preds, axis=0), index=y.index)
mae = np.mean(maes)
r2 = np.mean(r2s)
mape = np.mean(mapes)
r_pearson = np.mean(r_pearsons)

y_pred_std = pd.Series(np.std(y_preds, axis=0), index=y.index)
mae_std = np.std(maes)
r2_std = np.std(r2s)
mape_std = np.std(mapes)
r_pearson_std = np.std(r_pearsons)

y_pred_min = np.min(y_preds, axis=0)
y_pred_max = np.max(y_preds, axis=0)

# make the parity plot
plt.figure(figsize=(4.,4.))

index_new = df.index.tolist()
index_new = list(set(index_new).intersection(y.index))

plt.errorbar(y.loc[index_new], y_pred.loc[index_new], yerr=y_pred_std.loc[index_new], 
             fmt='.', alpha=0.6, markersize=13,
             ecolor='red',capsize=2)
if target == stability:
    plt.xlabel('Actual Ru dissolution (%)',fontsize=fontsize)
    plt.ylabel('Predicted Ru dissolution (%)',fontsize=fontsize)
# same x and y limits
ax = plt.gca()

if target == OP:
    ax.set_xlim([y.min()-25, y.max()+25])
    ax.set_ylim([y.min()-25, y.max()+25])
    # add a diagonal line
    ax.plot(ax.get_xlim(), ax.get_xlim(), ls="-", c=".1")
    ax.plot(ax.get_xlim(), np.array(ax.get_ylim())+20, ls=":", c=".1")
    ax.plot(ax.get_xlim(), np.array(ax.get_ylim())-20, ls=":", c=".1")

    # add the RMSE, MAE, and R2 to the plot
    plt.text(0.05, 0.95, f'MAE: {mae:.1f}'+r'$\pm$'+f'{mae_std:.1f}', transform=ax.transAxes, )
    plt.text(0.05, 0.9, r'$R_{pearson}$'+f': {r_pearson:.3f}'+r'$\pm$'+f'{r_pearson_std:.3f}', transform=ax.transAxes, )
elif target == stability:
    # use log scale for both axis
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([0.04, 10])
    ax.set_ylim([0.04, 10])
    # add a diagonal line
    ax.plot(ax.get_xlim(), ax.get_xlim(), ls="-", c=".1")
    ax.plot(ax.get_xlim(), [i*10 for i in ax.get_xlim()], ls=":", alpha=0.5, c=".1")
    ax.plot(ax.get_xlim(), [i/10 for i in ax.get_xlim()], ls=":", alpha=0.5, c=".1")
    ax.plot(ax.get_xlim(), [i*5 for i in ax.get_xlim()], ls="--", alpha=0.35, c=".1")
    ax.plot(ax.get_xlim(), [i/5 for i in ax.get_xlim()], ls="--", alpha=0.35, c=".1")
    # # add the MAPE, MAE, and R2 to the plot
    # plt.text(0.01, 0.95, f'MAPE: {mape:.1f}'+r'$\pm$'+f'{mape_std:.1f}%', transform=ax.transAxes, )
    # plt.text(0.01, 0.9, r'$R_{pearson}$'+f': {r_pearson:.3f}'+r'$\pm$'+f'{r_pearson_std:.3f}', transform=ax.transAxes, )
    print(f'MAPE: {mape:.1f}'+r'$\pm$'+f'{mape_std:.1f}%')
    print(r'$R_{pearson}$'+f': {r_pearson:.3f}'+r'$\pm$'+f'{r_pearson_std:.3f}')

plt.savefig(f'{modelname}.png',dpi=300,bbox_inches='tight')

df_ = y.to_frame()
df_[f'{target} predicted'] = y_pred
# insert y_pred_std after y_pred
df_.insert(1, f'{target} predicted std', y_pred_std)
df_['Formula']=df['Formula']
df_['Synthesis number'] = df.loc[df_.index,'Synthesis number']
df_[f'{target} predicted min'] = y_pred_min
df_[f'{target} predicted max'] = y_pred_max
# if target == OP:
#     df_.to_csv(f'csv/{modelname}_OP.csv')
# elif target == stability:
#     df_.to_csv(f'csv/{modelname}_{stability}.csv')




#%%
'''
Get the shap importance

'''
import shap

shap.initjs()

# rename the columns using dict
dict_rename = {
    'Anneal temp': 'Annealing $T$',
    'Annealing time (h)': 'Annealing Time',
    'H2O amount (mL)': 'H$_2$O Fraction',
    # 'area weighted G_pbx -1 -1.5': 'Pourbaix Energy',
    'area weighted G_pbx 0 1.5': 'Pourbaix Energy',
    'ion range oxidation state': 'Oxid. Range',
    'ion std_dev EN difference': r'StDev $\Delta \chi$',
    'OP (10mA/cm2)': 'Overpotential',
    'ion max ionic char': 'Max. Ionic Char.',
}
X = X.rename(columns=dict_rename)

#%%
y_preds = []
n_repeat = 10
shap_values = []
for random_state in range(n_repeat):
    pipeline = get_model(modelname=modelname,random_state=random_state)
    regressor = pipeline.named_steps['model'].regressor
    regressor.fit(X, np.log10(y))
    y_pred = regressor.predict(X)
    # convert the predicted values back to the original scale
    y_pred = 10**y_pred
    y_preds.append(y_pred)
    shap_values_all = shap.TreeExplainer(regressor).shap_values(X)
    shap_values.append(shap_values_all)


#%%
y_pred = np.mean(y_preds, axis=0)
y_pred_std = np.std(y_preds, axis=0)
# calculate the mean shap values
shap_values = np.mean(shap_values, axis=0)
# shap_values_all = pd.DataFrame(shap_values_all, columns=X.columns, index=X.index)

#%%
df_shap = pd.concat(
    [X, y, 
     pd.Series(
         y_pred,index=y.index,name=f'{target} predicted'
                     ), 
    pd.Series(y_pred_std,index=y.index,name=f'{target} predicted std')],
    axis=1,    
    )
# df_shap.to_csv(f'csv/{modelname}_pdp.csv')



#%%

shap.summary_plot(
    shap_values, 
    X,
    plot_type="dot",
    max_display=5,
    show=False,
                  )
# get the figure and axis
fig, ax = plt.gcf(), plt.gca()
# set the size of the figure
fig.set_size_inches(6, 4.5)
ax.set_xlabel('SHAP Value')
plt.tight_layout()
# plt.savefig('paper_figures/shap_plot.png', dpi=300)

# %%
