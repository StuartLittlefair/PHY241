import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fig, axis = plt.subplots(figsize=(12,10))



# Read csv file and define dataframe object (df for dataframe)
df = pd.read_csv('../data/gapminderDataFiveYear.txt', sep='\t')

# most recent data
year_mask = (df['year'] == 2007)
df_year = df[year_mask]

color_dict = dict(
    Asia='#1f77b4',
    Europe='#ff7f0e',
    Africa='#2ca02c',
    Americas='#d62728',
    Oceania='#9467bd'
)

max_population = df_year['pop'].max()
for name, color in color_dict.items():
    mask = (df_year['continent'] == name)
    x = df_year[mask]['gdpPercap']
    y = df_year[mask]['lifeExp']
    area = 3500*df_year[mask]['pop']/max_population
    axis.scatter(x,y,s=area,c=color,alpha=0.7,label=name)

lgnd = axis.legend(loc='upper left', scatterpoints=1, ncol=3)
for handle in lgnd.legendHandles:
    handle._sizes=[80]
    
    
axis.set_xscale('log')
axis.set_xlim(180,100000)
axis.set_ylim(40,90)
axis.set_xlabel('GDP per person')
axis.set_ylabel('Life Expectancy in Years')
plt.show()