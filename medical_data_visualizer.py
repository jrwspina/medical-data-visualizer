import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
def is_overweight(weight, height):
    bmi = weight /((height/100)**2)
    if bmi < 25.0:
        return 0
    return 1

df['overweight'] = df.apply(lambda x: is_overweight(x['weight'], x['height']), axis=1)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
def normalize(value):
    if value > 1:
        return 1
    return 0

df['cholesterol'] = df.apply(lambda x: normalize(x['cholesterol']), axis=1)
df['gluc'] = df.apply(lambda x: normalize(x['gluc']), axis=1)

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    columns = ['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    df_cat_melted = pd.melt(df, id_vars='cardio', value_vars=columns)
    df_cat_melted['total'] = 1

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cat = df_cat_melted.groupby(['cardio', 'variable', 'value'], as_index=False).count()

    # Draw the catplot with 'sns.catplot()'
    fig = sns.catplot(x='variable', y='total', hue='value', data=df_cat, col='cardio', kind='bar').fig

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig

# Draw Heat Map
def draw_heat_map():
    # Clean the data
    filter1 = df['ap_lo'] <= df['ap_hi']
    filter2 = df['height'] >= df['height'].quantile(0.025)
    filter3 = df['height'] <= df['height'].quantile(0.975)
    filter4 = df['weight'] >= df['weight'].quantile(0.025)
    filter5 = df['weight'] <= df['weight'].quantile(0.975)
    
    df_heat = df[filter1 & filter2 & filter3 & filter4 & filter5]

    # Calculate the correlation matrix
    corr = df_heat.corr(method='pearson')

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure

    fig, ax = plt.subplots(figsize=(14,10))

    # Draw the heatmap with 'sns.heatmap()'

    ax = sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", linewidths=1
                     ,center=0.0, vmax=0.25, vmin=-0.1, square=True, cbar_kws={"shrink":0.5})

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
