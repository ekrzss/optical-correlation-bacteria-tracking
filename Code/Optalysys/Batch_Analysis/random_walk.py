#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 14:53:36 2020

@author: erick
"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# np.random.seed(1234)

def brownian_motion(N, T, h):
    """
    Simulates a Brownian motion
    :param int N : the number of discrete steps
    :param int T: the number of continuous time steps
    :param float h: the variance of the increments
    """   
    dt = 1. * T/N  # the normalizing constant
    random_increments_x = np.random.normal(0.0, 1.0 * h, N)*np.sqrt(dt)  # the epsilon values
    brownian_motion_x = np.cumsum(random_increments_x)  # calculate the brownian motion
    brownian_motion_x = np.insert(brownian_motion_x, 0, 0.0) # insert the initial condition
    
    random_increments_y = np.random.normal(0.0, 1.0 * h, N)*np.sqrt(dt)  # the epsilon values
    brownian_motion_y = np.cumsum(random_increments_y)  # calculate the brownian motion
    brownian_motion_y = np.insert(brownian_motion_y, 0, 0.0) # insert the initial condition
    
    random_increments_z = np.random.normal(0.0, 1.0 * h, N)*np.sqrt(dt)  # the epsilon values
    brownian_motion_z = np.cumsum(random_increments_z)  # calculate the brownian motion
    brownian_motion_z = np.insert(brownian_motion_z, 0, 0.0) # insert the initial condition
    
    brownian_motion = np.stack((brownian_motion_x, brownian_motion_y, brownian_motion_z), axis=1)
    random_increments = np.stack((random_increments_x, random_increments_y, random_increments_z), axis=1)
    
    return brownian_motion, random_increments

def MSD(x, y, z, dt):
    
    msd = np.zeros(len(x))
    for i in range(1, len(x)):
        # msd[i] = np.mean((x[i:] - x[:len(x)-i])**2 + (y[i:] - y[:len(y)-i])**2 + (z[i:] - z[:len(z)-i])**2)
        msd[i] = np.mean((x[i:] - x[:-i])**2 + (y[i:] - y[:-i])**2 + (z[i:] - z[:-i])**2)
        # msd[i] = np.mean((x[i:] - x[:-i])**2)
        
    s = dt*np.arange(len(msd))
    
    return msd, s


N = 5000 # the number of discrete steps
T = 1 # the number of continuous time steps
dt = 1.0 * T/N  # total number of time steps
D = 0.5        # Diffuscion coefficient
h = 1 # the variance of the increments


# generate a brownian motion
X, epsilon = brownian_motion(N, T ,h)

MS_DISP, S = MSD(X[:, 0], X[:, 1], X[:, 2], dt)

#%% Plot

fig = plt.figure()

ax =  fig.add_subplot(1, 2, 1, projection='3d')
ax.plot(X[:, 0], X[:, 1], X[:, 2])
ax.scatter(X[0, 0], X[0, 1], X[0, 2], color='r', label='start', )
ax.scatter(X[-1, 0], X[-1, 1], X[-1, 2], color='y', label='end')
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
ax.set_title('Random Particle Trajectory: '+np.str(N)+' steps')
ax.legend(loc='best')


n_points = 200
ax = fig.add_subplot(1, 2, 2)
ax.plot(S[:n_points], MS_DISP[:n_points], '.-', label='MSD')
ax.plot(S[:n_points], 6*D*S[:n_points], label='6Dt')
ax.set_xlabel('t'); ax.set_ylabel('MSD')
ax.set_title('Mean Square Displacement: '+np.str(n_points)+' points')
ax.legend()
ax.grid()

plt.show()

#%%
    
# import plotly.express as px
# from plotly.offline import plot
# # df = px.data.iris()
# fig = px.line_3d(x=X[:,0], y=X[:,1], z=X[:,2])
# fig.show()
# plot(fig)

#%%

# import dash
# import dash_core_components as dcc
# import dash_html_components as html

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# app.layout = html.Div(children=[
#     html.H1(children='Hello Dash'),

#     html.Div(children='''
#         Dash: A web application framework for Python.
#     '''),

#     dcc.Graph(
#         id='example-graph',
#         figure={
#             'data': [
#                 {'x': X[:, 0], 'y': X[:, 1], 'z':X[:, 2], 'type': 'scatter3d', 'name': 'Trajectory', 'mode': 'lines'},
#                 {'x': X[0, 0], 'y': X[0, 1], 'z':X[0, 2], 'type': 'scatter3d', 'name': 'Start', 'mode':'marker', 'color':'yellow', 'size':221},
#                 {'x': X[-1, 0], 'y': X[-1, 1], 'z':X[-1, 2], 'type': 'scatter3d', 'name': 'End', 'mode':'marker', 'color':'red', 'size':221},
#             ],
#             'layout': {
#                 'title': 'Dash Data V'
#             }
#         }
#     )
# ])

# if __name__ == '__main__':
#     app.run_server(debug=True)
    