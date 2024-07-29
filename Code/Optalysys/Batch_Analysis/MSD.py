#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 17:22:07 2020

@author: erick
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def MSD(x, y, z, dt):
    
    msd = np.zeros(len(x))
    for i in range(1, len(x)):
        # msd[i] = np.mean((x[i:] - x[:len(x)-i])**2 + (y[i:] - y[:len(y)-i])**2 + (z[i:] - z[:len(z)-i])**2)
        msd[i] = np.mean((x[i:] - x[:-i])**2 + (y[i:] - y[:-i])**2 + (z[i:] - z[:-i])**2)
        # msd[i] = np.mean((x[i:] - x[:-i])**2)
        
    s = dt*np.arange(len(msd))
    #%
    ff = lambda x, c : c[0]*x + c[1] 
    coefs = np.polyfit(s[:100], msd[:100], 1)
    D = coefs[0]                                # Diffusion coefficient
    # fit = ff(s[:100], coefs)
    
    
    return msd, s, D

def main():

    step = 1
    size = 10000
    x = np.zeros(size)
    y = np.zeros(size)
    z = np.zeros(size)
    
    val = np.random.randint(1, 7, size=size)               # Uniform
    # val = np.random.binomial(5, 0.5, size=size) +1         # Binomial
    
    for i in range(1, len(x)):
        
        # val = np.random.randint(1, 7)
        # val = np.random.binomial(5, 0.5) +1
        # val = np.random.normal()
        
        if val[i] == 1: 
            x[i] = x[i - 1] + step
            y[i] = y[i - 1]
            z[i] = z[i - 1]
        elif val[i] == 2: 
            x[i] = x[i - 1] - step
            y[i] = y[i - 1]
            z[i] = z[i - 1]        
        elif val[i] == 3: 
            x[i] = x[i - 1] 
            y[i] = y[i - 1] + step
            z[i] = z[i - 1]
        elif val[i] == 4: 
            x[i] = x[i - 1] 
            y[i] = y[i - 1] - step
            z[i] = z[i - 1]
        elif val[i] == 5:
            x[i] = x[i - 1] 
            y[i] = y[i - 1]
            z[i] = z[i - 1] + step
        elif val[i] == 6:
            x[i] = x[i - 1] 
            y[i] = y[i - 1]
            z[i] = z[i - 1] - step
    
    dt = 0.5    
    msd, S, D = MSD(x, y, z, dt)
    
    fig = plt.figure()
    
    ax =  fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot(x, y, z)
    ax.scatter(x[0], y[0], z[0], color='r', label='start')
    ax.scatter(x[-1], y[-1], z[-1], color='y', label='end')
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.set_title('Random Particle Trajectory: '+np.str(size)+' steps')
    ax.legend(loc='best')
    
    n_points = 100
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(S[:n_points], msd[:n_points], '.-')
    ax.set_xlabel('t'); ax.set_ylabel('MSD')
    ax.set_title('Mean Square Displacement: '+np.str(n_points)+' points')
    ax.grid()
    
    plt.show()
    
    ##
    
    # import plotly.express as px
    # from plotly.offline import plot
    # # df = px.data.iris()
    # fig = px.line_3d(x=x, y=y, z=z)
    # fig.show()
    # plot(fig)
    
    ##
    
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
    #                 {'x': x, 'y': y, 'z':z, 'type': 'scatter3d', 'name': 'Trajectory', 'mode': 'lines'},
    #                 {'x': x[0], 'y': y[0], 'z':z[0], 'type': 'scatter3d', 'name': 'Start', 'mode':'marker', 'color':'red', 'size':221},
    #             ],
    #             'layout': {
    #                 'title': 'Dash Data Visualization'
    #             }
    #         }
    #     )
    # ])
    
    # if __name__ == '__main__':
    #     app.run_server(debug=True)
            
    # return 0

if __name__ == "__main__":
    main()
    