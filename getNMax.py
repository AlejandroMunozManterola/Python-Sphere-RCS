import numpy as np
import math
from DielectricMaterial import DielectricMaterial as DM
from src import *
from bessel import *

def getNMax(radius, sphere, background, frequency):
    '''
        determines the appropriate number of mie terms to evaluate
        Based on the wiscombe 1980 recommendation (which was deternined through 
        convergence behavior bessel functions at high orders that were
        calculated recursivelly).

        Designed to work for single-layered (monolithic) sphere. 
    '''
    #check that frequency input is correct  
    if (type(frequency) == int or type(frequency) == float):
        frequency = np.array([frequency])
    if (type(frequency) == list or type(frequency) == np.ndarray):
        frequency = np.array(frequency).flatten()
        M = len(frequency)
    else:
        print("wrong data type for frequency (in getNMax)")
    
    
    k_m = DM.getWaveNumber(background, frequency)
    x = abs(k_m * radius)
    #print(x)

    N_m = DM.getComplexRefractiveIndex(background, frequency)
    m = DM.getComplexRefractiveIndex(sphere, frequency) / N_m #relative refractive index

    N_max = np.ones((M,))
    for k in range(0,M):
        if (x[k] < 0.02):
            print("WARNING: it is better to use Rayleigh Scattering models for low frequencies.")
            print("\tNo less than 3 Mie series terms will be used in this calculation")
            #this comes from Wiscombe 1980: for size parameter = 0.02 or less, the number of terms
            #recommended will be 3 or less. 
            N_stop = 3
        elif (0.02 <= x[k] and x[k] <= 8):
            N_stop = x[k] + 4.*x[k]**(1/3) + 1
        elif (8 < x[k] and x[k] < 4200):
            N_stop = x[k] + 4.05*x[k]**(1/3) + 2
        elif (4200 <= x[k] and x[k] <= 20000):
            N_stop = x[k] + 4.*x[k]**(1/3) + 2
        else:
            print("WARNING: it is better to use Physical Optics models for high frequencies.")
            N_stop = 20000 + 4.*20000**(1/3) + 2
        
        #this is the KZHU original nmax formula (adapted for single sphere)
        #it recommends additional terms for high-index materials
        n_kzhu = abs(m[k] * x[k]) + 15

        # Cap N_max to prevent explosion for PEC-like materials (n >> 1)
        # The Wiscombe N_stop already ensures convergence; the KZHU term
        # is unnecessary for electrically large, highly conductive objects
        # since the Mie coefficients decay rapidly beyond N ~ x.
        N_max[k] = max(N_stop, min(n_kzhu, 500))

        #this is the Wiscombe-only implementation, seems to be accurate enough
        #N_max[k] = N_stop
        
    # Use max of Wiscombe N_stop values (ensures convergence for largest x)
    # but cap at 150 to avoid numerical instability of ric_bessely for
    # small arguments at high order (nu > ~x + 150 produces NaN)
    wiscombe_max = max([x[k] + 4.*x[k]**(1/3) + 2 if x[k] >= 8 else x[k] + 4.*x[k]**(1/3) + 1 if x[k] >= 0.02 else 3 for k in range(M)])
    N_final = min(int(math.ceil(wiscombe_max)), 150)
    return max(N_final, 3)


if __name__ == "__main__":
    radius = 0.5
    #sphere = DM(5,100)
    sphere = DM(2.56,0.5)
    background = DM(1,0)
    frequency = np.logspace(5,9,5)
    print(frequency)
    print(getNMax(radius, sphere, background, frequency))

    print(sphere.getComplexRefractiveIndex(1e9))

    