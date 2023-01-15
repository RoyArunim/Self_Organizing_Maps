# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 15:33:56 2023

@author: Dell
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv("D:/Masters/Self Organizing maps dataset/Self_Organizing_Maps/Credit_Card_Applications.csv")

X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
#splitting the X and Y not because it is supervised learning, but to store values if customer applications were approved or not
#no dependent variable



#Feature scaling

from sklearn.preprocessing import MinMaxScaler
sc= MinMaxScaler()
X=sc.fit_transform(X)


#Training the SOM
from minisom import MiniSom
som=MiniSom(10,10,15,sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train(X,100)

#Visualize the results
from pylab import bone, pcolor, colorbar, plot, show
bone() # just toinitialize the window
pcolor(som.distance_map().T) # mean inter neuron distance, transpose of that
colorbar()# for the legend of colors. the results show that the white boxes correspond to the highest inter neuron distance which means outliers and therefore frauds
'''
from here we can proceed, take the inverse transform of the winning nodes and see which nodes correspond to the frauds.
'''

'''
red circles customers who didnt get approval, green squares are customers who got approval
'''
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()
