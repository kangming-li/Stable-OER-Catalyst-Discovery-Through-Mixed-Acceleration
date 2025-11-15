#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kangming
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import chardet
import math 

#%% Get the phase information from excel file

xlsx = pd.read_excel('data/XRD Phaes analysis.xlsx').drop(
    columns=['exp No']
).set_index('XRD file')
xlsx = xlsx.iloc[:,:6]
xlsx.columns = ['Phase 1','PDF 1',
                'Phase 2','PDF 2',
                'Phase 3','PDF 3',]
phases = xlsx

#%% Get the XRD data from excel file
xlsx = pd.read_excel('data/Jade data.xlsx',sheet_name=None,skiprows=3)
# Access each sheet's dataframe using the sheet name as the key
dfs = {}
for sheet_name, df in xlsx.items():
    dfs[sheet_name] = df


#%%
def extract_info(text):
    # "(\w+) - Powder Diffraction" or "(\w+) - \(Unknown\)"
    # crystal_system_pattern = r"(\w+) - (Powder Diffraction|\(Unknown\))"
    crystal_system_pattern = r"(\w+) - .*mp="
    space_group_pattern = r",.*\((\d+)\).*mp="
    Z_pattern = r"Z=([0-9]+)\s*mp="
    density_c_pattern = r"Density\(c\)=([0-9.]+)"
    density_m_pattern = r"Density\(m\)=([0-9.]+)"
    mwt_pattern = r"Mwt=([0-9.]+)"
    vol_pattern = r"Vol=([0-9.]+)"

    # Search for patterns in the text
    crystal_system = re.search(crystal_system_pattern, text)
    space_group = re.search(space_group_pattern, text)
    Z = re.search(Z_pattern, text)
    density_c = re.search(density_c_pattern, text)
    density_m = re.search(density_m_pattern, text)
    mwt = re.search(mwt_pattern, text)
    vol = re.search(vol_pattern, text)

    # Extract and return the information
    return {
        "crystal_system": crystal_system.group(1) if crystal_system else None,
        "space_group": space_group.group(1) if space_group else None,
        "Z": int(Z.group(1)) if Z else None,
        "density_c": float(density_c.group(1)) if density_c else None,
        "density_m": float(density_m.group(1)) if density_m else None,
        "mwt": float(mwt.group(1)) if mwt else None,
        "vol": float(vol.group(1)) if vol else None
    }

def get_hkl(line,index=None):
    hkl = re.search(r"\( ([0-9]) ([0-9]) ([0-9])\)", line)
    if hkl is not None:
        h,k,l = hkl.groups()
        if index == 'h':
            return int(h)
        elif index == 'k':
            return int(k)
        elif index == 'l':
            return int(l)
        else:
            return int(h),int(k),int(l)        
    else:
        return None


def get_info_from_pdf_id(pdf_id,df):
    pdf_name = 'data/PDF data/PDF#'+str(pdf_id)+'.txt'

    # get the encoding of the pdf file
    with open(pdf_name, 'rb') as f:
        result = chardet.detect(f.read())   
    encoding = result['encoding']
    # read the pdf file
    with open(pdf_name, 'r', encoding=encoding) as f:
        text = f.read()

    # get the text below the line starting with "2-Theta"
    lines = text.split("\n")
    for i in range(len(lines)):
        if lines[i].startswith("2-Theta"):
            break
    lines_below = lines[i+1:]
    # remove the empty lines
    lines_below = [line for line in lines_below if line.strip() != ""]
    # match the pattern that contains "(num num num)" and assign it to hkl
    hkls = [get_hkl(line) for line in lines_below]

    # remove the pattern that contains "(num num num)"
    lines_below = [re.sub(r"\([0-9. ]+\)", "", line) for line in lines_below]
    lines_below = [line.strip() for line in lines_below]

    text_split = [re.split(r"\s+", line) for line in lines_below]
    text_split = [t_[:6] for t_ in text_split ]

    df_pdf = pd.DataFrame(
        # lines_below[1:], 
        # # use multiple spaces as separator
        # sep="\s+",
        # columns=lines_below[0].split()
        text_split,
        columns = ["2-Theta", "d(Å)", "I(f)", "Theta", "1/(2d)", "2pi/d"]
        )
    # get h
    df_pdf['h'] = [hkl[0] if hkl is not None else None for hkl in hkls]
    # get k
    df_pdf['k'] = [hkl[1] if hkl is not None else None for hkl in hkls]
    # get l
    df_pdf['l'] = [hkl[2] if hkl is not None else None for hkl in hkls]
    df_pdf['hkl'] = [str(hkl[0]) + str(hkl[1]) + str(hkl[2])
                     if hkl is not None else None for hkl in hkls]

    # replace the string in the 'I(f)' column with 0
    df_pdf['I(f)'] = df_pdf['I(f)'].replace('<1',0).astype(float)
    # sort the dataframe by the 'I(f)' column from large to small
    df_pdf = df_pdf.sort_values(by='I(f)',ascending=False)
    # get the first row
    theta2 = df_pdf.iloc[0,0]
    d= df_pdf.iloc[0,1]
    h,k,l = df_pdf.iloc[0,6],df_pdf.iloc[0,7],df_pdf.iloc[0,8]
    hkl = df_pdf.iloc[0,9]

    info = extract_info(text)
    info['2-Theta'] = float(theta2)
    info['h'] = h
    info['k'] = k
    info['l'] = l
    info['hkl'] = hkl

    # find the row in df whose '2-Theta' column is closest to theta2
    df['2-Theta'] = df['2-Theta'].astype(float)
    df['diff'] = (df['2-Theta'] - info['2-Theta']).abs()
    df = df.sort_values(by='diff')
    closest_row = df.iloc[0]
    info['height'] = closest_row['Height']
    info['area'] = closest_row['Area']
    info['fwhm'] = closest_row['FWHM']
    info['relative_intensity'] = closest_row['I%']
    # info['d'] = d

    return info #, closest_row

#%%
keys = ['crystal_system', 'space_group', 'Z', 
        'density_c', 'density_m', 'mwt', 'vol', 
        '2-Theta', 'height', 'area', 'fwhm', 
        'relative_intensity','h','k','l','hkl']

# a, b, c

#%%
new_df = pd.DataFrame()

# iterate over rows of the dataframe phases
for xrd_id, phase in phases.iterrows():
    xrd_id = str(xrd_id)
    # print(xrd_id)
    # # get the dataframe of the corresponding xrd_id
    if xrd_id in dfs:
        df = dfs[xrd_id]
    else:
        print(f"{xrd_id} is not found in Jade data.xlsx")
        continue
    
    try: 
        df = dfs[xrd_id]

        # create a dictionary to store the information
        new_dict = {}
        # iterate over the three phases
        for i in [1,2,3]:
            # get the pdf_id
            pdf_id = phase['PDF '+str(i)]
            # add the phase name to the dictionary
            new_dict['Phase '+str(i)] = phase['Phase '+str(i)]
            # if pdf_id is a string, get the information from the pdf file
            if isinstance(pdf_id, str):
                # info = get_info_from_pdf_id(pdf_id,df)
                info = get_info_from_pdf_id(pdf_id,df)
                # add the information to the dictionary
                for key in keys:
                    new_dict[key+' '+str(i)] = info[key]
            # if pdf_id is NaN, add None to the dictionary
            else:
                if math.isnan(pdf_id):
                    for key in keys:
                        new_dict[key+' '+str(i)] = None
                else:
                    raise ValueError(
                        f"pdf_id must be a string or NaN. (pdf_id={pdf_id},xrd_id={xrd_id})  {type(pdf_id)}"
                        )
        # add the dictionary to the dataframe, using pd.concat
        new_df = pd.concat([new_df,pd.DataFrame(new_dict,index=[xrd_id])])
            
    except Exception as e:
        print(e)

#%%

'''
What have been measured and what have not been measured?
How to preprocess?
'''

new_df = new_df.iloc[4:,]
# drop rows whose 'Phase 1' column is NaN
new_df = new_df.dropna(subset=['density_c 1'])

new_df.to_csv('xrd_features.csv')

# %%
