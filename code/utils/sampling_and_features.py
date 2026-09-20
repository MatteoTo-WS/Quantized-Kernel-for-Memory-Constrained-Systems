import numpy as np
import matplotlib.pyplot as plt

class SamplingFeatureClass():
    def __init__(self, M_RFF_dim, d_data_dim, sigma_ker, choice = "gaussian", nu = 1.5, ro = 1): # generic version requires :  kernel_f, pdf, a = -1, b = 1
        self.M = M_RFF_dim
        self.d = d_data_dim
        self.choice = choice
        self.nu = nu
        self.ro = ro
        # self.kernel_f = kernel_f
        # self.pdf = pdf
        # self.a = a
        # self.b = b
        self.u = np.random.uniform(0, 2 * np.pi, size=(1, self.M))
         
        if choice == "gaussian":
            self.w = np.random.normal(0, 1/sigma_ker, (self.M, self.d)) 

        elif choice == "matern":
            u_temp = np.random.chisquare(2 * self.nu, size=(self.M, 1))
            y_temp = np.random.normal(0, 1/(self.ro), (self.M, self.d))
            self.w = np.sqrt(2*self.nu / u_temp) * y_temp 
        
    # def inverse_cdf(self, t):
    #     samples = np.linspace(self.a, self.b, 10001)
    #     p_x = self.pdf(samples)
    #     cdf = np.cumsum(p_x) / np.sum(p_x)
    #     func_ppf = sp.interpolate.interp1d(cdf, samples, fill_value='extrapolate')
    #     return func_ppf(t)
    
    # def random_projection(self):
    #     u_uniform = np.random.uniform(0, 1, (self.M, self.d))   
    #     w = self.inverse_cdf(u_uniform)
    #     return w
    
    def feature_map_approx(self, x, mode = 1):    # x shape (n_samples, d), w shape (M, d)
        w = self.w
        x = np.atleast_2d(x)  # Ensure x is 2D
        w = np.atleast_2d(w)  # Ensure w is 2D

        if mode == 0: 
            z_cos_x = np.cos(x @ w.T).reshape(x.shape[0],w.shape[0])  # shape (n_samples, M)    <- Usually x is a vector, so (1,M)
            z_sin_x = np.sin(x @ w.T).reshape(x.shape[0],w.shape[0])  # shape (n_samples, M)
            
            z_bar_x = np.sqrt(1/(self.M)) * np.concatenate((z_cos_x, z_sin_x), axis = 1) # shape (n_samples, 2M)

        else:
            z_bar_x = np.sqrt(2/self.M) * np.cos(x @ w.T + self.u)

        return z_bar_x

    def sgn_feature_map(self, x, mode = 1):
        return np.sqrt(1/((2-mode)*self.M)) * np.sign(self.feature_map_approx(x, mode))
    
    def s_bit_feature_map(self, x, s, mode=1, random_quantization = False):
        w = self.w
        x = np.atleast_2d(x)
        w = np.atleast_2d(w)
        
        # standard features in [-1, 1]
        if mode == 0:
            raw_cos = np.cos(x @ w.T)
            raw_sin = np.sin(x @ w.T)
            raw_features = np.concatenate((raw_cos, raw_sin), axis=1)
            scale = np.sqrt(1 / self.M) 
        else:
            raw_features = np.cos(x @ w.T + self.u)
            scale = np.sqrt(2 / self.M)
   

        # 2. Quantize the features in [-1, 1] to 2^s levels
        L = 2**s # number of levels/endpoints

        # Map from [-1, 1] to [0, L - 1]
        mapped_features = (raw_features + 1) / 2 * (L - 1)
        
        # Round to nearest integer (nearest endpoint)
        if random_quantization == True: 
            quantized_indices = self.random_quantization_function(mapped_features)
        else:
            if s == 1:    # QUESTO NON è VERO (vedi paper Braking the waves) -> all aumentare di s abbiamo che il kernel quantizzato converge a quello approssimato standard che a sua volta converge, in media, a quello originale.
                scale = np.sqrt(1 / self.M)
            # ma possiamo dire che all aumentare di s le feature quantizzate sono "dense" rispetto alle feature randomiche standard, quidni manteniamo lo stesso scaling.
            quantized_indices = np.round(mapped_features)
        
        # Map back to [-1, 1]
        quantized_features = -1 + quantized_indices * (2 / (L - 1))
        
        return scale * quantized_features


    def random_quantization_function(self, x):
        x = np.atleast_2d(x)
        p = x % 1
        c = np.random.binomial(1, p, size = p.shape) 
        x = np.floor(x) + c
        return x


    # approx kernel function: 
    def approx_kernel(self, x, y, mode = 1):
        return self.feature_map_approx(x, mode) @ self.feature_map_approx(y, mode).T

    # quantized kernel function (1-bit / sign):
    def quantized_kernel(self, x, y, mode = 1):
        return self.sgn_feature_map(x, mode) @ self.sgn_feature_map(y, mode).T
        
    # s-bit quantized kernel function:
    def s_bit_quantized_kernel(self, x, y, s, mode = 1, random_quantization = False):
        return self.s_bit_feature_map(x, s, mode, random_quantization) @ self.s_bit_feature_map(y, s, mode, random_quantization).T
    
    # I am not sure about the soundness of these two functions, but I will implement them anyway. 
    # They are not used in the main notebook, so they can be easily removed if they are wrong.
    # Also, to apply these, we need to store the full-matrix in memory.

    def approx_kernel_matrix(self, x, mode = 1): 
        return self.feature_map_approx(x, mode) @ self.feature_map_approx(x, mode).T

    def quantized_kernel_matrix(self, x, mode = 1):
        return self.sgn_feature_map(x, mode) @ self.sgn_feature_map(x, mode).T
        
    def s_bit_quantized_kernel_matrix(self, x, s, mode = 1, random_quantization = False, fill = True):
        a = self.s_bit_feature_map(x, s, mode, random_quantization)
        # print(a)
        K = a @ a.T
        # if s == 1:
        if fill == True:
            np.fill_diagonal(K, 1.0)
        return K

