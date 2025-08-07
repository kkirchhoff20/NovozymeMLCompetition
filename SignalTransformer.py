import statsmodels.api as sm
import scipy
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.fft import fft, ifft
import numpy as np
import pywt

class AutoregressiveCoefficients(BaseEstimator, TransformerMixin):
    def __init__(self, order=4, start_index=0, end_index=8042):
        self.order = order
        self.start_index = start_index
        self.end_index = end_index
    
        
    def fit(self, X, y=None):
        """Need to enforce expected number of columns
        """
        return self
    
    def transform(self, X, y=None):
        # Perform arbitary transformation
        
        # Have to do these one by one?
        def coefficients(row):
            rho, sigma2 = sm.regression.linear_model.burg(row, order=self.order)
            return rho
        result = X.iloc[:, self.start_index:self.end_index].apply(coefficients, axis=1).to_numpy()
        return np.stack(result)

class WaveletPacketEntropy(BaseEstimator, TransformerMixin):
    def __init__(self, level=4, wavelet='haar', mode='symmetric', start_index=0, end_index=8042):
        self.level = level
        self.wavelet = wavelet
        self.mode = mode
        self.start_index = start_index
        self.end_index = end_index
    
    def fit(self, X, y=None):
        """Need to enforce expected number of columns
        """
        return self
    
    def transform(self, X, y=None):
        wp = pywt.WaveletPacket(data=X.iloc[:, self.start_index:self.end_index], wavelet=self.wavelet, mode=self.mode)
        node_entropies = []
        
        for leaf_node in wp.get_level(self.level):
            leaf_energies = np.square(np.abs(leaf_node.data))
            total_energy_at_level = np.sum(leaf_energies)
            prob = leaf_energies/total_energy_at_level
            entropy = scipy.stats.entropy(prob, axis=1)
            node_entropies.append(entropy)
            
        return np.vstack(node_entropies).T

class Normalizer(TransformerMixin, BaseEstimator):
    """
    Normalizes sample data to the interval (-1, 1)
    
    Parameters
    ----------
    
    """
    def __init__(self, nbits: int, signed: bool):
        self.signed = signed
        self.nbits = nbits
        if self.signed:
            self.nbits = self.nbits-1
    
    def transform(self, samples: np.array) -> np.array:
        normalized_samples = samples/(2**self.nbits) 
        normalized_samples = np.clip(normalized_samples, -1, 1)
        return normalized_samples
    
    def fit(self, X, y):
        return self

class NormFFT(TransformerMixin, BaseEstimator):    
    def __init__(self, n: int = 5000):
        self.n=n
        
    def transform(self, normalized_samples: np.array) -> (np.array, np.array, np.array):
        transformed_samples = fft(normalized_samples, n=self.n) # calculate fourier transform (complex numbers list)
        length = len(transformed_samples)/2  # you only need half of the fft list (real signal symmetry)
        halved_transform = transformed_samples[:, 0:int(length-1)]
        #return np.abs(halved_transform), np.real(halved_transform), np.imag(halved_transform)
        return np.abs(transformed_samples)
    
    def fit(self, X, y):
        return self