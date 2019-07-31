import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pylab as plt


class Hyperelastic:
    def __init__(self, model, parameters, order):
        self.model = model
        self.order = order
        self.parameters = parameters
        initialGuessMu = np.array([0.1]*self.order)
        initialGuessAlpha = np.array([0.2]*self.order)
        self.initialGuessParam = np.append(initialGuessMu,initialGuessAlpha)

        # Initialization of the Ogden model
        if model == 'Ogden':
            initialGuessMu = np.array([0.1]*self.order)
            initialGuessAlpha = np.array([0.2]*self.order)
            self.initialGuessParam = np.append(initialGuessMu,initialGuessAlpha)
            self.nbparam = self.order*2
        #elif ... Initialization of other models (not detailed here)
        else:
            print("Error")


    def OgdenModel(self, parameters, trueStrain):
        """Ogden hyperelastic model (incompressible material under uniaxial tension)
        Uses true strain and true stress data"""             
        # parameter is a 1D array: [mu0,mu1,...,mun,alpha0,alpha1,...,alphan]
        # get the mu and alpha parameters out of the 1D array
        muVec = parameters.reshape(2, self.order)[0]
        alphaVec = parameters.reshape(2, self.order)[1]
        lambd = np.exp(trueStrain)
        # broadcasting method to speed up computation
        lambd = lambd[np.newaxis, :]
        muVec = muVec[:self.order, np.newaxis]
        alphaVec = alphaVec[:self.order, np.newaxis]
        
        trueStress = np.sum(2*muVec*(lambd**(alphaVec-1) - lambd**(-((1/2)*alphaVec + 1))), axis=0)
        return trueStress


# Function to calculate the residuals.
# The fitting function holds the parameter values
def objectiveFun_Callback(parameters, exp_strain, exp_stress): 
    theo_stress = hyperelastic.ConsitutiveModel(parameters, exp_strain)   
    err = exp_stress - theo_stress
    return err

# read data from file
data = pd.read_csv('Ecoflex50', delimiter = ';',skiprows=9, names = ['Time (s)','True Strain','True Stress (MPa)','Engineering Strain','Engineering Stress (MPa)']) 
exp_strain = data['True Strain'].values    # converts panda series to numpy array
exp_stress = data['True Stress (MPa)'].values

# Instanciate a Hyperelastic object
hyperelastic = Hyperelastic('Ogden', np.array([0]), order=3)
    
# The least_squares package calls the Levenberg-Marquandt algorithm
# best-fit parameters are kept within optim_result.x
optim_result = least_squares(objectiveFun_Callback, hyperelastic.initialGuessParam, method ='lm', args=(exp_strain, exp_stress))
optim_parameters = optim_result.x


# Compute the true stress from the Ogden model with optimized parameters   
theo_stress = hyperelastic.ConsitutiveModel(optim_parameters, exp_strain)


# Plot experimental and predicted data on the same graph
plt.plot(exp_strain,exp_stress,'k',linewidth=2)
plt.plot(exp_strain,theo_stress,'r--', linewidth=2)