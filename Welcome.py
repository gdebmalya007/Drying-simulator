# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 14:13:03 2021

@author: dg668
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn as sklearn
from sklearn.model_selection import train_test_split
import datetime
import scipy as spint
import matplotlib as mpl
from numpy.random import default_rng
import matplotlib.colors as clr
from matplotlib import cm
import joblib
from tensorflow.keras import layers
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import streamlit as stlt
import os
stlt.title('High-fidelity simulation in a blink!')
dirpath=os.getcwd()

stlt.image(dirpath+'/Schematic.JPG')
stlt.write(' This appliction computes the change in temperature and moisture profiles in a 2d or 2d axisymmetric geometry for different product and process parameter values.' 
           ' With a total of 13 parameters available to change the possibilities of performing what-if scenarios are endless! '
            ' The model is based on a dynamic LSTM deep learning framework which is trained with the solution from a comprehensive multiphase and multiphysics based mechanistic model.')


