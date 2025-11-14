
from pymatgen.core.composition import Composition
from matminer.featurizers.conversions import StrToComposition, CompositionToOxidComposition
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.composition.ion import OxidationStates, IonProperty, ElectronAffinity, ElectronegativityDiff
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import math

def stoichiometry_featurizer(formulas):
    '''
    A featurizer that takes a list of chemical formula as inputs and returns a 
    list of features for each. If the list of chemical formula contains N elements,
    then the output should be a N x M matrix, where M is the number of features which is 
    the composition of elements in the formula.
    '''

    # Get the set of all elements present in the formulas
    all_elements = set()
    for formula in formulas:
        comp = Composition(formula)
        all_elements.update(comp.elements)


    # Initialize a dictionary to hold the features
    features = {str(element): [] for element in all_elements}

    # For each formula, get the composition of each element
    for formula in formulas:
        comp = Composition(formula)
        for element in all_elements:
            features[str(element)].append(comp.get_atomic_fraction(element))

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(features)

    return df


def get_oxid_from_formula(string):
    '''
    get the oxidation states from the formula
    '''
    comp = Composition(string)
    comp = comp.get_integer_formula_and_factor()[0]
    comp = Composition(comp)
    comp = CompositionToOxidComposition(max_sites=-1).featurize(comp)[0]
    return comp


# combine a list of dicts into one dict, by adding the values with the same key
def combine_dicts(list_of_dicts):
    '''
    combine a list of dicts into one dict, by adding the values with the same key
    '''
    dict_combined = {}
    for dict_ in list_of_dicts:
        for key, value in dict_.items():
            if key in dict_combined:
                dict_combined[key] += value
            else:
                dict_combined[key] = value
    return dict_combined

def get_cation(x):
    x = dict(x.as_dict())
    x = {key: value for key, value in x.items() if '+' in key}
    x = Composition(x)
    return x

def get_anion(x):
    x = dict(x.as_dict())
    x = {key: value for key, value in x.items() if '-' in key}
    x = Composition(x)
    return x


# def combine_precursors(df,n_precursors=4,concentrations=False):
#     ''' 
#     Get the combined precursor formula (Precursors 1 to 4)
#     '''
#     # iterate over the rows of df
#     composition = []
#     list_oxids = []
#     concentrations=[]
#     for index, row in df.iterrows():
#         string = ''
#         oxids=[]
#         concentration = 0
#         for i in range(1,n_precursors+1):
#             precursor = row['Precursor '+str(i)]
#             if isinstance(precursor, str):
#                 amount = row['amount of precursor '+str(i)+' (mmol)']
#                 string_ = '('+precursor+')'+str(amount)    
#                 # print(index, string_) # just for debugging        
#                 oxid_ = get_oxid_from_formula(string_)
#                 oxid_ = dict(oxid_)
#                 oxid_ = {key: value*amount for key, value in oxid_.items()}
#                 oxids.append(oxid_)
#                 concentration += amount
#                 string += string_
#         if concentrations:
#             concentration = concentration/(concentration+row['Solvent amount (mL)'])
#             concentrations.append(concentration)
#         oxids = combine_dicts(oxids)
#         oxids = Composition(oxids)
#         list_oxids.append(oxids)
#         comp_ = Composition(string)
#         composition.append(comp_)
#     # add the composition to df
#     df['Precursor combined'] = composition # composition of the combined precursors
#     df['oxids combined'] = list_oxids # oxidation states of all elements in the combined precursors
#     if concentrations:
#         df['concentrations'] = concentrations
#     df['cation combined'] = df['oxids combined'].apply(get_cation)
#     df['anion combined'] = df['oxids combined'].apply(get_anion)
#     return df



def process_row(row, n_precursors, concentrations):
    string = ''
    oxids=[]
    concentration = 0
    for i in range(1,n_precursors+1):
        precursor = row['Precursor '+str(i)]
        amount = row['amount of precursor '+str(i)+' (mmol)']

        if isinstance(precursor, str):

            if not math.isnan(amount) and amount>0:
                string_ = '('+precursor+')'+str(amount)    
                oxid_ = get_oxid_from_formula(string_)
                oxid_ = dict(oxid_)
                oxid_ = {key: value*amount for key, value in oxid_.items()}
                oxids.append(oxid_)
                concentration += amount
                string += string_

    if concentrations:
        concentration = concentration/(concentration+row['Solvent amount (mL)'])
        concentrations.append(concentration)

    oxids = combine_dicts(oxids)
    oxids = Composition(oxids)
    comp_ = Composition(string)

    return comp_, oxids, concentration



def apply_func(df, func):
    with mp.Pool(mp.cpu_count()) as pool:
        return list(tqdm(pool.imap(func, df), total=len(df)))

def combine_precursors(df,n_precursors=4,concentrations=False):
    ''' 
    Get the combined precursor formula (Precursors 1 to 4)
    '''
    # iterate over the rows of df
    with mp.Pool(mp.cpu_count()) as pool:
        results = list(
            tqdm(
                pool.imap(
                    partial(
                        process_row, 
                        n_precursors=n_precursors, 
                        concentrations=concentrations
                        ), 
                    [row for _, row in df.iterrows()]), 
                    total=len(df)
                    )
                    )
    composition, list_oxids, concentrations = zip(*results)

    # add the composition to df
    df['Precursor combined'] = composition # composition of the combined precursors
    df['oxids combined'] = list_oxids # oxidation states of all elements in the combined precursors
    if concentrations:
        df['concentrations'] = concentrations
    df['cation combined'] = apply_func(df['oxids combined'], get_cation)
    df['anion combined'] = apply_func(df['oxids combined'], get_anion)

    return df


def featurize_precursors(
        df,
        n_precursors=4,
        concentrations=False,
        ignore_errors=False
        ):

    '''
    Including more stats seem to make the model worse. strange
    '''
    # combining precursors
    print('combining precursors')
    df = combine_precursors(df,n_precursors=n_precursors,concentrations=concentrations)

    # combine the featurizers into one featurizer
    print('featurizing precursors')
    ion_featurizers = MultipleFeaturizer([
        OxidationStates(stats=["maximum", "minimum", "range", "std_dev"]), 
        IonProperty(fast=True), 
        ElectronAffinity(), 
        ElectronegativityDiff(stats=["maximum", "minimum", "range", "std_dev"]),
        ])
    ion_feature_labels = ion_featurizers.feature_labels()
    ion_features = ion_featurizers.featurize_dataframe(
        df, 
        col_id='oxids combined',
        ignore_errors=ignore_errors,
        )[ion_feature_labels]
    ion_features.columns = [f'ion {col}' for col in ion_features.columns]


    cation_featurizers = OxidationStates(stats=["maximum", "minimum", "range", "std_dev"])
    cation_feature_labels = cation_featurizers.feature_labels()
    cation_features = cation_featurizers.featurize_dataframe(
        df, 
        col_id='cation combined',
        ignore_errors=ignore_errors,
        )[cation_feature_labels]
    cation_features.columns = [f'cation {col}' for col in cation_features.columns]

    # same thing for anion
    anion_featurizers = OxidationStates(stats=["maximum", "minimum", "range", "std_dev"])
    anion_feature_labels = anion_featurizers.feature_labels()
    anion_features = anion_featurizers.featurize_dataframe(
        df, 
        col_id='anion combined',
        ignore_errors=ignore_errors,
        )[anion_feature_labels]
    anion_features.columns = [f'anion {col}' for col in anion_features.columns]
    return df,ion_features, cation_features, anion_features,




def plot_cm(y, y_pred, labels=[0,1], fontsize=15,text=None):
    # use plt to plot the confusion matrix, and add the values on the plot
    cm = confusion_matrix(y, y_pred,labels=labels)
    cm_normalized = confusion_matrix(y, y_pred,labels=labels, normalize='true')
    
    # use a light color map
    plt.imshow(cm, cmap=plt.cm.Blues)
    # get the current ax
    ax = plt.gca()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if (i,j) == (1,1):
                color = 'white'
            else:
                color = 'black'

            plt.text(j, i, cm[i,j], ha='center', va='center',fontsize=fontsize,color=color)
            plt.text(j, i+0.1, 
                     f'({cm_normalized[i,j]*100:.0f}%)', 
                     ha='center', va='center',
                     fontsize=fontsize,color=color)
    
    # set the ticks. Modify the labels to show 'gelable' and 'non-gelable' instead of '1' and '0'
    plt.yticks([0,1], labels)
    plt.xticks([0,1], labels)
    # set the ticks. Modify the labels to show 'gelable' and 'non-gelable' instead of '1' and '0'
    plt.yticks([0,1], ['non-gelable','gelable'],fontsize=fontsize-3)
    plt.xticks([0,1], ['non-gelable','gelable'],fontsize=fontsize-3)
    # set the fontsize of the labels in the axes
    plt.xlabel('Predicted label', fontsize=fontsize)
    plt.ylabel('True label', fontsize=fontsize)
    # plt.title('Confusion matrix', fontsize=fontsize)
    # set xtick in the upper x axis
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    # get fig and ax 
    fig = plt.gcf()
    ax = plt.gca()
    if text:
        ax.text(0.1, -0.1, text, fontsize=fontsize-2,transform=ax.transAxes)

    plt.tight_layout()
    plt.show()
    # return the figure
    return fig, ax