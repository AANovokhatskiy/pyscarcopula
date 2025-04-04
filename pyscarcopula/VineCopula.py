import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from pyscarcopula import ClaytonCopula, FrankCopula
from pyscarcopula.EllipticalCopula import BivariateGaussianCopula
from pyscarcopula import pobs

from typing import Literal
from pyscarcopula.stattests.gof_copula import get_smoothed_sample
from pyscarcopula.metrics.latent_process import latent_process_final_state

class VineCopula:
    def __init__(self, bivariate_classes = None, structure: Literal['cvine'] = 'cvine'):
        self.structure = structure
        self.bivariate_classes = [ClaytonCopula, FrankCopula, BivariateGaussianCopula] if bivariate_classes is None else bivariate_classes  # List of copula classes (e.g., [ClaytonCopula, FrankCopula, NormalCopula])
        self.allow_rotations = True

        self.thetas = None
        self.selected_copulas = None  # To store the best copula for each pair
        self.selected_rotations = None
        self.method = None
        self.fit_results = None

    def choose_best_copula(self, u1, u2, **kwargs):
        """
        Choose the best bivariate copula for the given pair of variables (u1, u2).
        
        Parameters:
            u1 (np.ndarray): First variable (shape: (T,)).
            u2 (np.ndarray): Second variable (shape: (T,)).
        
        Returns:
            best_copula: The best copula object.
            best_theta: The fitted parameter for the best copula.
            best_log_likelihood: The log-likelihood of the best copula.
        """
        best_copula = None
        best_theta = None
        best_rotation = None
        best_log_likelihood = -np.inf
        best_fit_result = None

        # Try each copula and select the one with the highest log-likelihood
        tempu = np.column_stack((u1, u2))

        for copula_class in self.bivariate_classes:
            if self.allow_rotations == True:
                if copula_class().rotatable == True:
                    rotations_list = [0, 90, 180, 270]
                else:
                    rotations_list = [0]
            else:
                rotations_list = [0]

            for angle in rotations_list:
                copula = copula_class(dim = 2, rotate = angle)
                # copula.rotate = angle
                fit_result = copula.fit(data = tempu, to_pobs = False)
                theta = fit_result.x[0] if len(fit_result.x) == 1 else fit_result.x
                log_likelihood = fit_result.fun
            
                if log_likelihood > best_log_likelihood:
                    best_copula = copula
                    best_theta = theta
                    best_log_likelihood = log_likelihood
                    best_rotation = angle
                    best_fit_result = fit_result
        
        if self.method.upper() != 'MLE':
            fit_result = copula.fit(data = tempu, to_pobs = False, **kwargs)
            best_theta = fit_result.x[0] if len(fit_result.x) == 1 else fit_result.x
            best_log_likelihood = fit_result.fun
            best_fit_result = fit_result

        return best_fit_result, best_copula, best_theta, best_log_likelihood, best_rotation

    def fit_c_vine_copula(self, u, **kwargs):
        n, d = u.shape
        
        method = kwargs.get("method", 'mle')
        self.method = method

        self.thetas = np.zeros((d - 1, d - 1), dtype=object)
        self.selected_copulas = np.empty((d - 1, d - 1), dtype=object)
        self.selected_rotations = np.empty((d - 1, d - 1), dtype=object)
        self.fit_results = np.empty((d - 1, d - 1), dtype=object)

        v = np.zeros((d, d, n))
        
        # Initialize the first row of v with the input data
        for i in range(d):
            v[0, i, :] = u[:, i]
        
        # Iterate over the tree levels
        for j in range(d - 1):
            # Fit the conditional copulas for the current tree level
            for i in range(d - j - 1):
                u1 = v[j, 0, :]  # First variable
                u2 = v[j, i + 1, :]  # Second variable
                best_copula = None
                best_theta = None
                best_log_likelihood = -np.inf
                
                # Try each copula and select the one with the highest log-likelihood
                fit_result, best_copula, best_theta, best_log_likelihood, best_rotation = self.choose_best_copula(u1, u2, **kwargs)

                
                # Store the best copula and its parameter
                self.thetas[j, i] = best_theta
                self.selected_copulas[j, i] = best_copula
                #self.selected_copulas[j, i].rotate = best_rotation
                self.selected_rotations[j, i] = best_rotation
                self.fit_results[j, i] = fit_result
            
            # Stop if we have processed all tree levels
            if j == d - 2:
                break
            
            # Compute the values for the next tree level using the h-function
            for i in range(d - j - 1):
                tempu = np.column_stack((v[j, 0, :], v[j, i + 1, :]))
                
                r = get_smoothed_sample(self.selected_copulas[j, i], tempu, self.fit_results[j, i])
                v[j + 1, i, :] = self.selected_copulas[j, i].h(v[j, i + 1, :], v[j, 0, :], r) #self.thetas[j, i]
        
        return self.thetas
    
    def c_vine_log_likelihood(self, u, **kwargs):
        if self.thetas is None or self.selected_copulas is None:
            raise ValueError("Fit the copula first using fit_c_vine_copula")
        
        n, d = u.shape
        log_likelihood = 0.0
        v = np.zeros((d, d, n))
        
        # Initialize the first row of v with the input data
        for i in range(d):
            v[0, i, :] = u[:, i]
        
        # Iterate over the tree levels
        for j in range(d - 1):
            # Compute the log-likelihood contribution for the current tree level
            for i in range(d - j - 1):
                u1 = v[j, 0, :]
                u2 = v[j, i + 1, :]
                tempu = np.column_stack((u1, u2))
                alpha = np.asarray(self.thetas[j, i])
                log_likelihood += -self.selected_copulas[j, i].mlog_likelihood(alpha, tempu,
                                                                               method = self.method,
                                                                              **kwargs)
            
            # Stop if we have processed all tree levels
            if j == d - 2:
                break
            
            # Compute the values for the next tree level
            for i in range(d - j - 1):
                tempu = np.column_stack((v[j, 0, :], v[j, i + 1, :]))
                r = get_smoothed_sample(self.selected_copulas[j, i], tempu, self.fit_results[j, i])

                v[j + 1, i, :] = self.selected_copulas[j, i].h(v[j, i + 1, :], v[j, 0, :], r) #self.thetas[j, i]
        
        return log_likelihood
    
    def sample_c_vine_copula(self, n):
        if self.thetas is None or self.selected_copulas is None:
            raise ValueError("Fit the copula first using fit_c_vine_copula")
        
        d = self.thetas.shape[0] + 1
        w = np.random.uniform(0, 1, (n, d))
        x = np.zeros((n, d))
        v = np.zeros((d, d, n))
        
        x[:, 0] = w[:, 0]
        v[0, 0, :] = w[:, 0]
        
        for i in range(1, d):
            v[i, 0, :] = w[:, i]
            for k in range(i - 1, -1, -1):
                # u1 = v[k, 0, :]  # Первая переменная
                # u2 = v[k, i - k, :]  # Вторая переменная
                # tempu = np.column_stack((u1, u2))
                tempu = x

                r = get_smoothed_sample(self.selected_copulas[k, i - k - 1], tempu, self.fit_results[k, i - k - 1])

                v[i, 0, :] = self.selected_copulas[k, i - k - 1].h_inverse(v[i, 0, :], v[k, k, :], r) #self.thetas[k, i - k - 1]
            x[:, i] = v[i, 0, :]
            
            if i == d - 1:
                break
            
            for j in range(i):
                # tempu = np.column_stack((v[j, 0, :], v[j, i - j, :]))
                tempu = x
                
                r = get_smoothed_sample(self.selected_copulas[j, i - j - 1], tempu, self.fit_results[j, i - j - 1])

                v[i, j + 1, :] = self.selected_copulas[j, i - j - 1].h(v[i, j, :], v[j, j, :], r) #self.thetas[j, i - j - 1]
        
        return x

    def predict_c_vine_copula(self, n):
        if self.thetas is None or self.selected_copulas is None:
            raise ValueError("Fit the copula first using fit_c_vine_copula")
        
        d = self.thetas.shape[0] + 1
        w = np.random.uniform(0, 1, (n, d))
        x = np.zeros((n, d))
        v = np.zeros((d, d, n))
        
        x[:, 0] = w[:, 0]
        v[0, 0, :] = w[:, 0]
        
        for i in range(1, d):
            v[i, 0, :] = w[:, i]
            for k in range(i - 1, -1, -1):               
                r = latent_process_final_state(self.thetas[k, i - k - 1], 
                                                self.method, 
                                                n)
                if self.method.upper() != 'MLE':
                    r = self.selected_copulas[k, i - k - 1].transform(r)

                v[i, 0, :] = self.selected_copulas[k, i - k - 1].h_inverse(v[i, 0, :], v[k, k, :], r) #self.thetas[k, i - k - 1]
            x[:, i] = v[i, 0, :]
            
            if i == d - 1:
                break
            
            for j in range(i):      
                r = latent_process_final_state(self.thetas[j, i - j - 1], 
                                                self.method, 
                                                n)
                if self.method.upper() != 'MLE':
                    r = self.selected_copulas[k, i - k - 1].transform(r)      

                v[i, j + 1, :] = self.selected_copulas[j, i - j - 1].h(v[i, j, :], v[j, j, :], r) #self.thetas[j, i - j - 1]
        
        return x

    def fit(self, u, to_pobs = False, allow_rotations = True, **kwargs):
        self.allow_rotations = allow_rotations

        if self.structure == 'cvine':
            _u = np.copy(np.asarray(u, dtype = np.float64))
            if to_pobs == True:
                _u = pobs(_u)
            return self.fit_c_vine_copula(_u, **kwargs)
        else:
            raise ValueError(f"Unknown structure: {self.structure}")

    def mlog_likelihood(self, u, **kwargs):
        if self.thetas is None:
            raise ValueError("Fit the copula first using fit method")
            
        if self.structure == 'cvine':
            return -self.c_vine_log_likelihood(u, **kwargs)
        else:
            raise ValueError(f"Unknown structure: {self.structure}")

    def get_sample(self, size):
        if self.thetas is None:
            raise ValueError("Fit the copula first using fit method")
        
        if self.structure == 'cvine':
            if self.method.upper() != 'MLE':
                raise NotImplementedError("sampling implemented only for mle")
            return self.sample_c_vine_copula(size)
        else:
            raise ValueError(f"Unknown structure: {self.structure}")

    def get_predict(self, size):
        if self.thetas is None:
            raise ValueError("Fit the copula first using fit method")
        
        if self.structure == 'cvine':
            return self.predict_c_vine_copula(size)
        else:
            raise ValueError(f"Unknown structure: {self.structure}")