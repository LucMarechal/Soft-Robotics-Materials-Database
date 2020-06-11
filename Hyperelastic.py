# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 22:31:21 2019

@author: marechlu
"""

import numpy as np

class Hyperelastic:

    def __init__(self, model, parameters, order):
        self.model = model
        self.order = order
        self.parameters = parameters
        self.param_names = []

        if model == 'Ogden':
            initialGuessMu = np.array([0.1]*self.order)
            initialGuessAlpha = np.array([0.2]*self.order)
            self.initialGuessParam = np.append(initialGuessMu,initialGuessAlpha)
            self.nbparam = self.order*2
            muVec_names = ["µ1","µ2","µ3"][0:self.order]
            alphaVec_names = ["α1","α2","α3"][0:self.order]
            self.param_names = np.append(muVec_names,alphaVec_names)
        elif model == 'Neo Hookean':
            self.initialGuessParam = np.array([0.1])
            self.nbparam = 1            
            self.param_names = ["µ"]
        elif model == 'Yeoh':
            self.initialGuessParam = np.array([0.1]*self.order)
            self.nbparam = self.order
            self.param_names = ["C1","C2","C3"][0:order] 
        elif model == 'Mooney Rivlin':
            self.initialGuessParam = np.array([0.1]*self.order)
            self.nbparam = self.order
            self.param_names = ["C10","C01","C20"][0:self.order]
        elif model == 'Gent':
            self.initialGuessParam = np.array([0.1]*2)
            self.nbparam = 2
            self.order=2
            self.param_names = ["µ","Jm"]
        elif model == 'Veronda Westmann':
            self.initialGuessParam = np.array([0.1]*2)
            self.nbparam = 2
            self.order=2
            self.param_names = ["C1","C2"] 
        elif model == 'Humphrey':
            self.initialGuessParam = np.array([0.1]*2)
            self.nbparam = 2
            self.order=2
            self.param_names = ["C1","C2"]            
        else:
            print("Error")




    def YeohModel(self, cVec, trueStrain):
        """Yeoh hyperelastic model (incompressible material under uniaxial tension)
        Uses true strain and true stress data"""
    
        lambd = np.exp(trueStrain)
        I1 = lambd**2 + 2/lambd

        trueStress = np.zeros((self.order,len(trueStrain)))
        for i in range (0,self.order):

            trueStress[i,:] = 2*(lambd**2 - 1/lambd)*i*cVec[i]*(I1-3)**(i-1)
            trueStress_sum = np.sum(trueStress, axis=0)
        return trueStress_sum



    def NeoHookeanModel(self, mu, trueStrain):
        """Neo-Hookean hyperelastic model (incompressible material under uniaxial tension)
        Uses true strain and true stress data"""
    
        lambd = np.exp(trueStrain)   # lambd i.e lambda
        trueStress = 2*mu*(lambd**2 - 1/lambd)
        return trueStress



    def OgdenModel(self, parameters, trueStrain):
        """Ogden hyperelastic model (incompressible material under uniaxial tension)
        Uses true strain and true stress data"""
                
        # parameter is a 1D array : [mu0,mu1,...,mun,alpha0,alpha1,...,alphan] 
        muVec = parameters.reshape(2, self.order)[0]
        alphaVec = parameters.reshape(2, self.order)[1]
        lambd = np.exp(trueStrain)
        # broadcasting method to speed up computation
        lambd = lambd[np.newaxis, :]
        muVec = muVec[:self.order, np.newaxis]
        alphaVec = alphaVec[:self.order, np.newaxis]
        
        trueStress = np.sum(2*muVec*(lambd**(alphaVec - 1) - lambd**(-((1/2)*alphaVec + 1))), axis=0)
        return trueStress



    def MooneyRivlinModel(self, cVec, trueStrain):
        """Mooney Rivlin hyperelastic model (incompressible material under uniaxial tension)
        Uses true strain and true stress data"""
        
        cVec = np.append(cVec, np.zeros(3-self.order) ) #To ensure CXX is zero if unsed
        C10 = cVec[0]
        C01 = cVec[1]
        C20 = cVec[2]
        lambd = np.exp(trueStrain)        
        trueStress = 2*(lambd**2 - 1/lambd)*(C10 + C01/lambd + 2*C20*(lambd**2 + 2/lambd -3))
        return trueStress



    def GentModel(self, parameters, trueStrain):
        """Gent hyperelastic model (incompressible material under uniaxial tension)
        Uses true strain and true stress data"""
       
        mu = parameters[0]
        Jm = parameters[1]
        lambd = np.exp(trueStrain)
        I1 = lambd**2 + 2/lambd     
        trueStress = (lambd**2 - 1/lambd)*(mu*Jm / (Jm - I1 + 3))
        #trueStress = mu*Jm / (Jm - lambd**2 - 2/lambd + 3) * (lambd**2 - 1/lambd)
        return trueStress
 
    
    
    def VerondaWestmannModel(self, parameters, trueStrain):
        """Veronda-Westmann hyperelastic model (incompressible material under uniaxial tension)
        Uses true strain and true stress data"""    
        
        C1=parameters[0]
        C2=parameters[1]
        lambd = np.exp(trueStrain)    

        I1 = lambd**2 + 2/lambd        
        trueStress = 2*(lambd**2 - 1/lambd) * C1*C2*(np.exp(C2*(I1-3) - 1/(2*lambd)))
        return trueStress    
   
 
    def HumphreyModel(self, parameters, trueStrain):
        """Humphrey hyperelastic model (incompressible material under uniaxial tension)
        Uses true strain and true stress data"""    
        
        C1=parameters[0]
        C2=parameters[1]
        lambd = np.exp(trueStrain)    

        I1 = lambd**2 + 2/lambd        
        trueStress = 2*(lambd**2 - 1/lambd) * C1*C2*(np.exp(C2*(I1-3)))
        return trueStress        
    
    
    def ConsitutiveModel(self, parameters, trueStrain):
        """ Constitutive Model"""      
        
        self.parameters = parameters # update parameters attribute      
        
        if self.model == 'Ogden':
            trueStress = self.OgdenModel(self.parameters, trueStrain)
        elif self.model == 'Neo Hookean':
            trueStress = self.NeoHookeanModel(self.parameters, trueStrain)      
        elif self.model == 'Yeoh':
            trueStress = self.YeohModel(self.parameters, trueStrain)            
        elif self.model == 'Mooney Rivlin':
            trueStress = self.MooneyRivlinModel(self.parameters, trueStrain)            
        elif self.model == 'Gent':
            trueStress = self.GentModel(self.parameters, trueStrain) 
        elif self.model == 'Veronda Westmann':
            trueStress = self.VerondaWestmannModel(self.parameters, trueStrain) 
        elif self.model == 'Humphrey':
            trueStress = self.HumphreyModel(self.parameters, trueStrain) 
        else:
            print("Error")
            
        return trueStress