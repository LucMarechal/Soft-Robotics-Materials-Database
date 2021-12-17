# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 22:31:21 2019

@author: marechlu
"""

import numpy as np

class Hyperelastic:

    def __init__(self, model, parameters, order, data_type):
        self.model = model
        self.order = order
        self.parameters = parameters
        self.param_names = []
        self.data_type = data_type

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
            self.param_names = ["C1","C2","C3"][0:self.order] 
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




    def YeohModel(self, cVec, Strain):
        """Yeoh hyperelastic model (incompressible material under uniaxial tension)"""
    
        if self.data_type == 'True':
            lambd = np.exp(Strain)
        elif self.data_type == 'Engineering':
            lambd = 1 + Strain
        else:
            print("Data type error. Data is neither 'True' or 'Engineering'. ")

        I1 = lambd**2 + 2/lambd

        Stress = np.zeros((self.order,len(Strain)))

        for i in range (0,self.order):
            if self.data_type == 'True':
                Stress[i,:] = 2*(lambd**2 - 1/lambd)*(i+1)*cVec[i]*((I1-3)**(i)) # true
            elif self.data_type == 'Engineering':
                Stress[i,:] = 2*(lambd - 1/(lambd**2))*(i+1)*cVec[i]*(I1-3)**(i) # eng
            else:
                print("Data type error. Data is neither 'True' or 'Engineering'. ")

            Stress_sum = np.sum(Stress, axis=0)
        return Stress_sum



    def NeoHookeanModel(self, mu, Strain):
        """Neo-Hookean hyperelastic model (incompressible material under uniaxial tension)"""

        if self.data_type == 'True':
            lambd = np.exp(Strain)       # lambd i.e lambda
            Stress = mu*(lambd**2 - 1/lambd)
        elif self.data_type == 'Engineering':
            lambd = 1 + Strain
            Stress = mu*(lambd - 1/(lambd**2))
        else:
            print("Data type error. Data is neither 'True' or 'Engineering'. ")

        return Stress



    def OgdenModel(self, parameters, Strain):
        """Ogden hyperelastic model (incompressible material under uniaxial tension)"""
                
        # parameter is a 1D array : [mu0,mu1,...,mun,alpha0,alpha1,...,alphan] 
        muVec = parameters.reshape(2, self.order)[0]
        alphaVec = parameters.reshape(2, self.order)[1]
        
        if self.data_type == 'True':
            lambd = np.exp(Strain)       # lambd i.e lambda
        elif self.data_type == 'Engineering':
            lambd = 1 + Strain
        else:
            print("Data type error. Data is neither 'True' or 'Engineering'. ") 
        
        # broadcasting method to speed up computation
        lambd = lambd[np.newaxis, :]
        muVec = muVec[:self.order, np.newaxis]
        alphaVec = alphaVec[:self.order, np.newaxis]

        if self.data_type == 'True':
            Stress = np.sum(2*muVec*(lambd**(alphaVec - 1) - lambd**(-((1/2)*alphaVec + 1))), axis=0)
        elif self.data_type == 'Engineering':
            Stress = np.sum((2*muVec*(lambd**(alphaVec - 1) - lambd**(-((1/2)*alphaVec + 1)))/lambd), axis=0)        

        return Stress



    def MooneyRivlinModel(self, cVec, Strain):
        """Mooney Rivlin hyperelastic model (incompressible material under uniaxial tension)"""
        
        cVec = np.append(cVec, np.zeros(3-self.order) ) #To ensure CXX is zero if unsed
        C10 = cVec[0]
        C01 = cVec[1]
        C20 = cVec[2]

        if self.data_type == 'True':
            lambd = np.exp(Strain)       # lambd i.e lambda
            Stress = 2*(lambd**2 - 1/lambd)*(C10 + C01/lambd + 2*C20*(lambd**2 + 2/lambd -3))
        elif self.data_type == 'Engineering':
            lambd = 1 + Strain
            Stress = (2*(lambd**2 - 1/lambd)*(C10 + C01/lambd + 2*C20*(lambd**2 + 2/lambd -3)))/lambd
        else:
            print("Data type error. Data is neither 'True' or 'Engineering'. ")

        return Stress



    def GentModel(self, parameters, Strain):
        """Gent hyperelastic model (incompressible material under uniaxial tension)"""
       
        mu = parameters[0]
        Jm = parameters[1]

        if self.data_type == 'True':
            lambd = np.exp(Strain)       # lambd i.e lambda
        elif self.data_type == 'Engineering':
            lambd = 1 + Strain
        else:
            print("Data type error. Data is neither 'True' or 'Engineering'. ")   

        I1 = lambd**2 + 2/lambd

        if self.data_type == 'True':
            Stress = (lambd**2 - 1/lambd)*(mu*Jm / (Jm - I1 + 3))
        elif self.data_type == 'Engineering':
            Stress = (lambd - 1/lambd**2)*(mu*Jm / (Jm - I1 + 3))

        return Stress
 
    
    
    def VerondaWestmannModel(self, parameters, Strain):
        """Veronda-Westmann hyperelastic model (incompressible material under uniaxial tension)"""  
        
        C1=parameters[0]
        C2=parameters[1]

        if self.data_type == 'True':
            lambd = np.exp(Strain)       # lambd i.e lambda
        elif self.data_type == 'Engineering':
            lambd = 1 + Strain
        else:
            print("Data type error. Data is neither 'True' or 'Engineering'. ")   

        I1 = lambd**2 + 2/lambd

        if self.data_type == 'True':
            Stress = 2*(lambd**2 - 1/lambd) * C1*C2*(np.exp(C2*(I1-3) - 1/(2*lambd)))
        elif self.data_type == 'Engineering':
            Stress = (2*(lambd**2 - 1/lambd) * C1*C2*(np.exp(C2*(I1-3) - 1/(2*lambd))))/lambd                
        
        return Stress
   
 
    def HumphreyModel(self, parameters, Strain):
        """Humphrey hyperelastic model (incompressible material under uniaxial tension)"""   
        
        C1=parameters[0]
        C2=parameters[1]

        if self.data_type == 'True':
            lambd = np.exp(Strain)       # lambd i.e lambda
        elif self.data_type == 'Engineering':
            lambd = 1 + Strain
        else:
            print("Data type error. Data is neither 'True' or 'Engineering'. ")

        I1 = lambd**2 + 2/lambd        
        
        if self.data_type == 'True':
            Stress = 2*(lambd**2 - 1/lambd) * C1*C2*(np.exp(C2*(I1-3)))
        elif self.data_type == 'Engineering':
            Stress = 2*(lambd - 1/lambd**2) * C1*C2*(np.exp(C2*(I1-3)))

        return Stress        
    
    
    def ConsitutiveModel(self, parameters, Strain):
        """ Constitutive Model"""      
        
        self.parameters = parameters # update parameters attribute      
        
        if self.model == 'Ogden':
            Stress = self.OgdenModel(self.parameters, Strain)
        elif self.model == 'Neo Hookean':
            Stress = self.NeoHookeanModel(self.parameters, Strain)      
        elif self.model == 'Yeoh':
            Stress = self.YeohModel(self.parameters, Strain)            
        elif self.model == 'Mooney Rivlin':
            Stress = self.MooneyRivlinModel(self.parameters, Strain)            
        elif self.model == 'Gent':
            Stress = self.GentModel(self.parameters, Strain) 
        elif self.model == 'Veronda Westmann':
            Stress = self.VerondaWestmannModel(self.parameters, Strain) 
        elif self.model == 'Humphrey':
            Stress = self.HumphreyModel(self.parameters, Strain) 
        else:
            print("Error")
            
        return Stress
