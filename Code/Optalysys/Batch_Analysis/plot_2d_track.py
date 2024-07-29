# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 17:17:33 2021

@author: eers500
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import easygui as gui

path = gui.fileopenbox()
#%%
pnumber = 0
dataf = pd.read_csv(path)
dataf = dataf[dataf['TRACK_ID'] == pnumber]
track = dataf[['POSITION_Y', 'POSITION_X', 'POSITION_T']].values

plt.figure(5)
plt.plot(track[:, 1], -track[:, 0])
plt.axis('tight')
plt.title('SPT 2D Track')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

#%%
import seaborn as sns

data = pd.read_csv(path)
pnumber = p_number = np.unique(data['TRACK_ID'].values)

# for pn in pnumber:
    # track = data[['POSITION_Y', 'POSITION_X', 'POSITION_T', 'TRACK_ID']]
    # plt.scatter(track['POSITION_X'], -track['POSITION_Y'], c=track['TRACK_ID'])
    # sns.scatterplot(track['POSITION_X'], -track['POSITION_Y'], c=track['TRACK_ID'])
    
sns.scatterplot(data=data, x='POSITION_X', y='POSITION_Y', hue='TRACK_ID')

#%%
import plotly.express as px
from plotly.offline import plot, iplot


data = pd.read_csv(path)
pnumber = p_number = np.unique(data['TRACK_ID'].values)

fig = px.scatter(x = data['POSITION_X'], y = -data['POSITION_Y'], color=data['TRACK_ID'])
fig.show()
plot(fig)