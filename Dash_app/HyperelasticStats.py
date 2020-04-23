# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 17:06:46 2019

@author: Luc Marechal
USMB : Université Savoie Mont Blanc
"""
import numpy as np

class HyperelasticStats:
    
    def __init__(self, Yexp, Ymodel, p):
        self.target = Yexp
        self.model = Ymodel
        self.p = p
        self._n = len(Yexp)   


    def sse(self):
        '''returns sum of squared errors (model vs actual)'''
        squared_errors = (self.target - self.model) ** 2
        return np.sum(squared_errors)
        
    def sst(self):
        '''returns total sum of squared errors (actual vs avg(actual))'''
        avg_Yexp = np.mean(self.target)
        squared_errors = (self.target - avg_Yexp) ** 2
        return np.sum(squared_errors)
    
    def rmse(self):
        '''returns rmse'''
        squared_errors = (self.target - self.model) ** 2
        return np.sqrt(np.mean(squared_errors))

    def r_squared(self):
        '''returns calculated value of r^2'''
        return 1 - self.sse()/self.sst()
    
    def adj_r_squared(self):
        '''returns calculated value of adjusted r^2
        R¯² = 1 - (n-1)(R²-1)/(n-p-1)'''
        adj_squared_errors =   1 - ((self._n - 1)*(self.r_squared() - 1)/(self._n - self.p - 1))     
        return adj_squared_errors
    
    def aic(self):
       '''returns the Akaike Information Criterion'''   
       return self._n *np.log(self.sse()/self._n) + 2*self.p
   
    def S(self):
       '''returns the Residual Standard Error (S)'''   
       return np.sqrt(self.sse()/(self._n - self.p - 1))
   
    def mapd(self):
       '''returns the mean absolute percentage deviation (MAPD)'''
       relativeError = np.zeros(np.min([len(self.model),len(self.target)]))
       for i in range (0,np.min([len(self.model),len(self.target)])):
           relativeError[i] = np.absolute((self.target[i]-self.model[i])/self.model[i])
       return (100/self._n)*np.sum(relativeError)