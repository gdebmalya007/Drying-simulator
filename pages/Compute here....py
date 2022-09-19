# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 11:53:19 2021

@author: dg668
"""

import pandas as pd
import numpy as np
import os
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
import time

dirpath=os.getcwd()
stlt.title('Drying simulator')
with stlt.sidebar:
    avarch=stlt.checkbox('advanced variables')
with stlt.form('Data input'):
    with stlt.sidebar:
        pres=stlt.slider(r'Gauge pressure (bar)',-0.9,0.9,0.0,0.05)
        Ti=stlt.slider('Initial temperature (C)', 7, 47, 27, 1)+273
        Tb=stlt.slider(r'Drying temperature (C)', 37, 127, 82, 1)+273
        Xo=stlt.number_input('Initial moisture content (db)',2.0,5.0,3.5,0.1)
        rh=stlt.number_input('RH (%)',0.0,10.0,1.0,0.1)
        ht=stlt.slider('Heat transfer coeff (W/m^2K)',25,75,40,1)
        por=stlt.number_input('Porosity',0.75,0.95,0.88,0.01)
        kp=stlt.number_input('Thermal conductivity', 0.1,0.8,0.2,0.05)
        cp=stlt.number_input('Specific heat (J/kg K)',750,2500,1650,50)
        rhop=stlt.number_input('Density',800,2000,1650,10)-2
        indop=stlt.radio('Shape of the specimen',('Cuboid','Cylindrical dice'))
        if indop=='Cuboid':
            indbin=0
        else:
            indbin=1
        wi= stlt.slider('Width(cm)', 2.0,6.0,4.0,0.1)
        th=stlt.slider('Thickness(cm)',0.5,2.5,1.5,0.1)
        if avarch:
            stlt.warning('Change the variables')
            k_evap=stlt.slider('Constant of evaporation',100,1200,730,50)
            dv=stlt.number_input(label='Binary diffusion coefficient of vapor in air',
                        min_value=0.5,max_value=10.0,value=5.2,
                        step=0.1)*1e-5
        else:
            k_evap=1000
            dv=5.2e-5
        stlt.form_submit_button('solve')
satn=((rhop*(1-por)*Xo)/(0.018))*0.018/(por*998.2)
if satn > 0.95 or satn < 0.35:
    stlt.warning( "too much or too little water content: change porosity, initial moisture content or solid density")
    stlt.stop()
if Tb< Ti+15:
    stlt.warning("Very low drying temperature: select a higher drying temperature")
    stlt.stop()
if wi<th+1:
    stlt.warning("Aspect ratio out of bounds: select a lower thickness or a higher width")
    stlt.stop()
presi=pres*100000+101325
presb=pres*100000+101325
rhob= (610.7*np.power(10,(7.5*(Tb-273.15)/(Tb-35.85)))*rh/100*0.018)/(8.314*Tb)   
xv=(610.7*np.power(10,(7.5*(Ti-273.15)/(Ti-35.85))))/presi
hm=0.5*ht*dv*np.power((0.025/(1.225*1.01e3*dv)),0.33)/0.025
rng=default_rng()
tpapnp=joblib.load(dirpath+'/timestp.pckl')
tstpch=np.sort(rng.choice(tpapnp,600,replace=False))
grddat=pd.read_csv(dirpath+'/griddat.csv',usecols=[0,1],skiprows=0,header=None)
inrowsingle=np.array([presi,Ti,Xo,Tb,presb,rhob,ht,hm,k_evap,dv,por,kp,cp,rhop,indbin,
                      wi,th,xv]).reshape(-1,18)
inmatcont=np.repeat(np.repeat(inrowsingle,600,axis=0).reshape(-1,600,18),800,axis=0)
tstprow=np.repeat(np.array(tstpch).reshape(1,600,1),800,axis=0)
griddatx=np.repeat(np.array(grddat[0]).reshape(800,1,1),600,axis=1) 
griddaty= np.repeat(np.array(grddat[1]).reshape(800,1,1),600,axis=1) 
spattimin=np.append(tstprow,np.append(griddatx,griddaty,axis=2),axis=2) 
redinnet=np.append(inmatcont,spattimin,axis=2)

class NDStandardScaler(TransformerMixin):
    def __init__(self, **kwargs):
        self._scaler = StandardScaler(copy=True, **kwargs)
        self._orig_shape = None

    def fit(self, X, **kwargs):
        X = np.array(X)
        # Save the original shape to reshape the flattened X later
        # back to its original shape
        if len(X.shape) > 1:
            self._orig_shape = X.shape[1:]
        X = self._flatten(X)
        self._scaler.fit(X, **kwargs)
        return self

    def transform(self, X, **kwargs):
        X = np.array(X)
        X = self._flatten(X)
        X = self._scaler.transform(X, **kwargs)
        X = self._reshape(X)
        return X

    def _flatten(self, X):
        # Reshape X to <= 2 dimensions
        if len(X.shape) > 2:
            n_dims = np.prod(self._orig_shape)
            X = X.reshape(-1, n_dims)
        return X

    def _reshape(self, X):
        # Reshape X back to it's original shape
        if len(X.shape) >= 2:
            X = X.reshape(-1, *self._orig_shape)
        return X
    def inverse(self, X, **kwargs):
        X = np.array(X)
        X = self._flatten(X)
        X = self._scaler.inverse_transform(X, **kwargs)
        X = self._reshape(X)
        return X
scalerld=NDStandardScaler()
scalerld=joblib.load(dirpath+'/scalernew.pckl')
redinnet_sc= scalerld.transform(redinnet)
#@stlt.cache(suppress_st_warning=True,allow_output_mutation=True)

@stlt.experimental_singleton
def loadmods():
    firmod1=tf.keras.models.load_model(dirpath+'/modelfull1stack')
    #firmod1._make_predict_function()
    firmod2=tf.keras.models.load_model(dirpath+'/modelfull2stack')
    #firmod2._make_predict_function()
    firmod3=tf.keras.models.load_model(dirpath+'/modelfull3stack')
    #firmod3._make_predict_function()
    modstackT=tf.keras.models.load_model(dirpath+'/try1')
    #modstackT._make_predict_function()
    modstackTi=tf.keras.models.load_model(dirpath+'/try3imp')
    #modstackTi._make_predict_function()
    modstackcl=tf.keras.models.load_model(dirpath+'/try1cl')
    #modstackcl._make_predict_function()
    time.sleep(1)
    return firmod1, firmod2, firmod3, modstackT, modstackTi, modstackcl
with stlt.spinner('Loading the models....Should be faster from the second attempt'):
    mod1,mod2,mod3,stackT,stackT2,stackcl=loadmods()
stlt.success('Computing solution...')
modout1=mod1.predict(redinnet_sc)
modout2=mod2.predict(redinnet_sc)
modout3=mod3.predict(redinnet_sc)
modinstack=redinnet[:,:,18:21]
modfinalT=stackT.predict([modout1[:,:,0:1],modout2[:,:,0:1],modout3[:,:,0:1],modinstack])
modfinalcl=stackcl.predict([modout1[:,:,1:2],modout2[:,:,1:2],modout3[:,:,1:2],modinstack])




maxTval=joblib.load(dirpath+'/maxv')
minTval=joblib.load(dirpath+'/minv')
modfinalT_nsc=modfinalT*(maxTval-minTval)+minTval
modoutT1_nsc=modout1[:,:,0:1]*(maxTval-minTval)+minTval
modoutT2_nsc=modout2[:,:,0:1]*(maxTval-minTval)+minTval
modoutT3_nsc=modout3[:,:,0:1]*(maxTval-minTval)+minTval
stpltT,stpltT1,stpltT2,stpltT3=[],[],[],[]
for abc in range(800):
    stpltT.append(modfinalT_nsc[abc,:,0])
    stpltT1.append(modoutT1_nsc[abc,:,0])
    stpltT2.append(modoutT2_nsc[abc,:,0])
    stpltT3.append(modoutT3_nsc[abc,:,0])
stpltTC=np.array(stpltT)*(Tb-Ti)+Ti
stpltT1C=np.array(stpltT1)*(Tb-Ti)+Ti
stpltT2C=np.array(stpltT2)*(Tb-Ti)+Ti
stpltT3C=np.array(stpltT3)*(Tb-Ti)+Ti
stpltcl,stpltcl1,stpltcl2,stpltcl3=[],[],[],[]
for abc in range(800):
    stpltcl.append(modfinalcl[abc,:,0])
    stpltcl1.append(modout1[:,:,1:2][abc,:,0])
    stpltcl2.append(modout2[:,:,1:2][abc,:,0])
    stpltcl3.append(modout3[:,:,1:2][abc,:,0])
stpltclxm=np.array(stpltcl)*Xo
stpltclxm1=np.array(stpltcl1)*Xo
stpltclxm2=np.array(stpltcl2)*Xo
stpltclxm3=np.array(stpltcl3)*Xo



figpnt,axespnt=plt.subplots(nrows=1,ncols=2,figsize=(12,5))
pltpntT=axespnt[0].plot(tstpch,stpltTC[40*9+19]-273,'r^',markersize=3,label='Ensemble')
pltpntT1=axespnt[0].plot(tstpch,stpltT1C[40*9+19]-273,'b*',markersize=3, label='Best interval model')
pltpntcl=axespnt[1].plot(tstpch,stpltclxm[40*9+19]*Xo*(1-por)*rhop/0.018,'r^',markersize=3,label='Ensemble')
pltpntcl1=axespnt[1].plot(tstpch,stpltclxm1[40*9+19]*Xo*(1-por)*rhop/0.018,'bo',markersize=3,label='Best interval model')
axespnt[0].set_xlabel(r'Time (hr)')
axespnt[0].set_ylabel(r'Temperature $\left(^oC\right)$')
axespnt[0].set_title('Mid point temperature vs time')
axespnt[1].set_xlabel(r'Time (hr)')
axespnt[1].set_ylabel(r'Moisture concentration $\left(\frac{mol}{m^3}\right)$')
axespnt[1].set_title('Mid point water concentration vs time')
axespnt[0].legend()
axespnt[1].legend()
stlt.header('Plots at the midpoint of the domain')
stlt.pyplot(figpnt)


griddimx=np.array(grddat[0]*wi/4).reshape(20,40)
griddimy=np.array(grddat[1]*th).reshape(20,40)
ijk=rng.choice(np.arange(200,600),size=1)
contpltT=stpltTC[:,ijk].reshape(20,40)-273
contpltcl=stpltclxm[:,ijk].reshape(20,40)*Xo*(1-por)*rhop/0.018
figconT,axesconT=plt.subplots(nrows=1,ncols=1,figsize=(8,4))
figconcl,axesconcl=plt.subplots(nrows=1,ncols=1,figsize=(8,4))
#=clr.Normalize(vmax=max([max(contpltT.flat)]),vmin=min([min(contpltT.flat)]))
#cmapT = cm.get_cmap('hot', 10)
#mapT = cm.ScalarMappable(cmap=cmapT, norm=normlimT)
Tcon= axesconT.contourf(griddimx,griddimy,contpltT)
cbarT=figconT.colorbar(Tcon)
#plt.colorbar()
#cbarT.set_ticks(np.linspace(max([max(contpltT.flat)]),min([min(contpltT.flat)]),10))
#normlimcl=clr.Normalize(vmax=max([max(contpltcl.flat)]),vmin=min([min(contpltcl.flat)]))
#cmapcl = cm.get_cmap('bone', 10)
#mapcl = cm.ScalarMappable(cmap=cmapcl, norm=normlimcl)
clcon=axesconcl.contourf(griddimx,griddimy,contpltcl)
cbarcl=figconcl.colorbar(clcon)
axesconcl.set_title ('Moisture profile at'+ str(tstpch[ijk])+'hrs')
axesconT.set_title('Temperature profile at'+ str(tstpch[ijk])+'hrs')
#plt.colorbar()
#cbarcl.set_ticks(np.linspace(max([max(contpltcl.flat)]),min([min(contpltcl.flat)]),10))
stlt.header('Spatial profiles') 
stlt.pyplot(figconT)
stlt.pyplot(figconcl)
Tavmat,clavmat=[],[]
Tval,clval=[],[]
simp=spint.integrate.simps
for jj in range(600):
    Tavmat.append(stpltTC[:,jj].reshape(20,40))
    clavmat.append(stpltclxm[:,jj].reshape(20,40))
for fgh in range(600):
    Tval.append(simp(simp(Tavmat[fgh],griddimx[0,:],axis=1),griddimy[:,0])/(wi*(39/8000)*th*(19/2000)))
    clval.append(simp(simp(clavmat[fgh],griddimx[0,:],axis=1),griddimy[:,0])/(wi*(39/8000)*th*(19/2000)))
figavplt,axesav=plt.subplots(nrows=1,ncols=2,figsize=(12,5))
avpltT=axesav[0].plot(tstpch,np.array(Tval)-273,'b^',markersize=4)
axesav[0].set_xlabel(r'Time(hr)')
axesav[0].set_ylabel(r'Temperature $\left(^oC\right)$')
axesav[0].set_title('Average temperature vs time')
avpltcl=axesav[1].plot(tstpch,np.array(clval),'bo',markersize=4)
axesav[1].set_xlabel(r'Time(hr)')
axesav[1].set_ylabel(r'Moisture content (db)')
axesav[1].set_title('Moisture content(db) vs time')

stlt.header('Average plots')
stlt.pyplot(figavplt)
grdnld=np.append(np.array(grddat[0]*wi/4).reshape(800,1),np.array(grddat[1]*th).reshape(800,1),axis=1)
tstpchdnld=np.append(np.zeros((1,2)),np.array(tstpch).reshape(1,600), axis=1)
stpltTCdnld=np.append(tstpchdnld,np.append(grdnld,stpltTC,axis=1),axis=0)
stpltclxmdnld=np.append(tstpchdnld,np.append(grdnld,stpltclxm,axis=1),axis=0)
Tdnld=pd.DataFrame(stpltTCdnld-273.15).to_csv().encode('utf-8')
cldnld=pd.DataFrame(stpltclxmdnld*Xo*(1-por)*rhop/0.018).to_csv().encode('utf-8')
stlt.download_button(label="Download detailed solution for temperature", data= Tdnld, file_name='Soln_T.csv', mime='text/csv')
stlt.download_button(label="Download detailed solution for moisture concentration", data= cldnld, file_name='Soln_cl.csv', mime='text/csv')
#stackcl.summary(print_fn=lambda x: stlt.text(x))


#stlt.write(pd.DataFrame(stpltTC))