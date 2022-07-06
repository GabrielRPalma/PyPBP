# Importanting the packages
from .pypbp import *
import numpy as np
import emd

# Main functions
# Example
cdef int a, b 
cpdef Cadd(a, b):
    
    return(np.log(a)+b)

# Decomposition function
cdef list time_series
def get_decomposition(time_series):
            
    time_series_array = np.array(time_series)
    IMFs = emd.sift.sift(np.log(time_series_array+1))    

    return(np.exp(IMFs-1))
