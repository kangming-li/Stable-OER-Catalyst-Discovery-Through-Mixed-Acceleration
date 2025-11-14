#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from catalysis_yang_project import read_synthesis_data
from pymatgen.core import Composition, Element

#%%
df = read_synthesis_data(
    filename='data/metal dissolution-20250506.csv',
    skiprows=1,
    )


df['composition'] = df['Formula'].apply(lambda x: Composition(x))
df['elements'] = df['Formula'].apply(
    lambda x: [str(i) for i in Composition(x).elements]
    )
# create a df_success by droping the rows in df with 0 in 'Synthesis'
df_success = df[df['Synthesis'] != 0]


all_elements = set()
for i in df['elements']:
    all_elements.update(i)
# sort the elements by atomic number
all_elements = sorted(all_elements, key=lambda x: Element(x).Z)

# find how many times each element appears
element_count = {}
for i in all_elements:
    element_count[i] = 0
    for j in df['elements']:
        if i in j:
            element_count[i] += 1

element_count = pd.Series(element_count)
# element_count = element_count.sort_values(ascending=False)
element_count = element_count.reindex(sorted(element_count.index, key=lambda x: Element(x).Z))

# drop 'Ru'
element_count = element_count.drop('Ru')
# drop entries with 0 synthesis
element_count = element_count[element_count > 0]

# find how many times each element appears in the successful synthesis
element_count_success = {}
for i in all_elements:
    element_count_success[i] = 0
    for j in df_success['elements']:
        if i in j:
            element_count_success[i] += 1

element_count_success = pd.Series(element_count_success)
# element_count_success = element_count_success.sort_values(ascending=False)
element_count_success = element_count_success.reindex(sorted(element_count_success.index, key=lambda x: Element(x).Z))
# drop 'Ru'
element_count_success = element_count_success.drop('Ru')
# # drop entries with 0 successful synthesis
# element_count_success = element_count_success[element_count_success > 0]


element_count_success = element_count_success.loc[element_count.index]


#%%

def polar_bar_plot(heights, colors):
    nelements = len(heights)
    angles = np.linspace(0, 2 * np.pi, nelements, endpoint=False).tolist()
    variable_A = heights  # Bar heights (Variable A)
    variable_B = colors  # Bar colors (Variable B normalized between 0 and 1)

    # Normalize variable_B for color mapping
    norm = plt.Normalize(variable_B.min(), variable_B.max())
    # cmap = plt.cm.RdBu  # Red-Blue colormap
    cmap = plt.cm.coolwarm
    colors = cmap(norm(variable_B))

    # Plotting
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={'projection': 'polar'})

    # Bar width
    width = np.pi / nelements*1.95  # Adjust width as needed

    # Create bars
    bars = ax.bar(angles, variable_A, width=width, color=colors, edgecolor='none', alpha=1)
    # Set the labels
    ax.set_xticks(angles)
    ax.set_xticklabels(heights.index)

    ax.set_ylim(0, variable_A.max() * 1.01)  # Set the maximum limit for the radius
    ax.set_yticklabels([])
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))  # Adjust the number of radial ticks
    # change grid line width
    lw = 0.6
    ax.grid(linewidth=lw, linestyle=':')
    # Add a colorbar to show the mapping of Variable B
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Empty array to fix colorbar
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.1)
    cbar.set_label('Number of synthesis')

    # Add labels and format
    ax.set_theta_zero_location('N')  # Set 0° at the top
    ax.set_theta_direction(-1)  # Clockwise direction

    plt.show()


polar_bar_plot(element_count_success/element_count,element_count)


# %%
