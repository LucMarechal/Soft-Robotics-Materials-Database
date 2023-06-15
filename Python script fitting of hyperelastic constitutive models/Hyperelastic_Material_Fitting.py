import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from scipy.optimize import least_squares
from scipy.optimize import NonlinearConstraint
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize


class Hyperelastic:
    def __init__(self, model, parameters, order):
        self.model = model
        self.order = order
        self.parameters = parameters
        self.param_names = []
        self.fitting_method = 'lm'

        # Initialization of the Ogden model
        if model == 'Ogden':
            initialGuessMu = np.array([1.0]*self.order)
            initialGuessAlpha = np.array([1.0]*self.order)
            self.initialGuessParam = np.append(initialGuessMu,initialGuessAlpha)
            self.nbparam = self.order*2
            muVec_names = ["µ1","µ2","µ3"][0:self.order]
            alphaVec_names = ["α1","α2","α3"][0:self.order]
            self.param_names = np.append(muVec_names,alphaVec_names)
            self.fitting_method = 'trust-constr'
        elif model == 'Neo Hookean':
            self.initialGuessParam = np.array([0.1])
            self.nbparam = 1            
            self.param_names = ["µ"]
            self.fitting_method = 'lm'
        elif model == 'Mooney Rivlin':
            self.initialGuessParam = np.array([0.1]*self.order)
            self.nbparam = self.order
            self.param_names = ["C10","C01","C20"][0:self.order]
            self.fitting_method = 'trust-constr'    
        #elif ... Initialization of other models (not detailed here but the Hyperelastic class is avalailbe on the Github repo)
        else:
            print("Error. Wrong name of model in Hyperelastic")


    def OgdenModel(self, parameters, trueStrain):
        """Ogden hyperelastic model (incompressible material under uniaxial tension)
        Uses true strain and true stress data"""
                
        # parameter is a 1D array : [mu0,mu1,...,mun,alpha0,alpha1,...,alphan] 
        muVec = parameters[0:self.order]              # [mu0,mu1,...,mun]
        alphaVec = parameters[self.order:]            # [alpha0,alpha1,...,alphan]
        lambd = np.exp(trueStrain)
 
        # broadcasting method to speed up computation
        lambd = lambd[np.newaxis, :]
        muVec = muVec[:self.order, np.newaxis]
        alphaVec = alphaVec[:self.order, np.newaxis]
        
        trueStress = np.sum(muVec*(lambd**alphaVec - 1/(lambd**(alphaVec/2))), axis=0)
        return trueStress 


# Function to calculate the residuals.
# The fitting function holds the parameter values
def objectiveFun_Callback(parameters, exp_strain, exp_stress):
    theo_stress = hyperelastic.ConsitutiveModel(parameters, exp_strain)   
    # The cost function for Levenberg-Marquardt and Trust Constraint algorithms are not expressed the same way ! Check Scipy documentation 
    if hyperelastic.fitting_method == 'lm':
        residuals = theo_stress - exp_stress
    elif hyperelastic.fitting_method == 'trust-constr':
        residuals = np.sqrt(sum((theo_stress-exp_stress)**2.0))        
    else:
        print("Error, please chose either 'lm' or 'trust-constr' as fitting method")
    
    return residuals

# read data from file
data = pd.read_csv('Ecoflex50', delimiter = ';',skiprows=9, names = ['Time (s)','True Strain','True Stress (MPa)','Engineering Strain','Engineering Stress (MPa)']) 
exp_strain = data['True Strain'].values        # converts panda series to numpy array
exp_stress = data['True Stress (MPa)'].values

# Instanciate a Hyperelastic object
hyperelastic = Hyperelastic('Ogden', np.array([0]), order=3)
    
# The least_squares package calls the Levenberg-Marquandt algorithm
# best-fit parameters are kept within optim_result.x
if hyperelastic.fitting_method == 'trust-constr':   
    if hyperelastic.model == 'Ogden':
        # Non Linear Conditions for the Ogden model : mu0*alpha0 > 0, mu1*alpha1 > 0, mu2*alpha2 > 0,
        const = NonlinearConstraint(hyperelastic.NonlinearConstraintFunction, 0.0, np.inf, jac=hyperelastic.NonlinearConstraintJacobian)#, hess='2-point')
    elif hyperelastic.model == 'Mooney Rivlin':
        # Linear Conditions for the Mooney Rivlin model : C10 + C01 > 0
        const = LinearConstraint([[1.0, 1.0, 0.0][0:hyperelastic.order], [0.0, 0.0, 0.0][0:hyperelastic.order]], 0.0, np.inf)
    else:
        const=()
        
    # The ogden and Mooney Rivlin models need constraint optimisation which cannot be done with the Levenberg-Marquandt algorithm
    optim_result = minimize(objectiveFun_Callback, hyperelastic.initialGuessParam, args=(exp_strain, exp_stress), method='trust-constr', constraints=const, tol=1e-12)    
elif hyperelastic.fitting_method == 'lm':
    # The least_squares package calls the Levenberg-Marquandt algorithm.
    # best-fit paramters are kept within optim_result.x
    optim_result = least_squares(objectiveFun_Callback, hyperelastic.initialGuessParam, method ='lm', gtol=1e-12, args=(exp_strain, exp_stress))   
else:
    print("Error in fitting method")

optim_parameters = optim_result.x


# Compute the true stress from the Ogden model with optimized parameters   
theo_stress = hyperelastic.ConsitutiveModel(optim_parameters, exp_strain)


# Plot experimental and predicted data on the same graph
plt.plot(exp_strain,exp_stress,'k',linewidth=2)
plt.plot(exp_strain,theo_stress,'r--', linewidth=2)
