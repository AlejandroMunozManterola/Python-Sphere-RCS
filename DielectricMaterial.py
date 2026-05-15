import numpy as np

class DielectricMaterial:
    ''' Translation of KZHU Dielectric material Class
        G. Kevin Zhu (2021). Sphere scattering MATLAB Central File Exchange
        (https://www.mathworks.com/matlabcentral/fileexchange/31119-sphere-scattering)
    '''
    

    def __init__(
        self,
        epsilon_r,
        sigma_e,
        mu_r=1,
        sigma_m=0,
        name="",
        epsilon_r_complex=None,
        loss_tangent=None,
    ):
        '''
            epsilon_r:  dielectric permittivity of material in steady state
            sigma_e:    electronic conductivity of material
            mu_r:       relative magnetic permeability in steady state
                        (assumed to be 1)
            sigma_m:    magnetic conductivity?
                        (assumed to be 0)
            epsilon_r_complex:
                        optional complex relative permittivity. If provided,
                        the dispersive part is taken directly from this value.
            loss_tangent:
                        optional dielectric loss tangent tan(delta).
                        Uses epsilon_r * (1 - j*tan(delta)).

            Notes:
                - Provide either epsilon_r_complex OR loss_tangent OR sigma_e
                  for electric losses. Mixing these can double-count losses and
                  is not allowed.
        '''
        if epsilon_r_complex is not None and loss_tangent is not None:
            raise ValueError("Use either epsilon_r_complex or loss_tangent, not both.")

        if epsilon_r_complex is not None and sigma_e != 0:
            raise ValueError("sigma_e must be 0 when epsilon_r_complex is provided.")

        if loss_tangent is not None and sigma_e != 0:
            raise ValueError("sigma_e must be 0 when loss_tangent is provided.")

        if np.iscomplexobj(epsilon_r) and epsilon_r_complex is None:
            epsilon_r_complex = complex(epsilon_r)
            epsilon_r = float(np.real(epsilon_r_complex))

        self.epsilon_r = float(np.real(epsilon_r))
        self.epsilon_r_complex = None if epsilon_r_complex is None else complex(epsilon_r_complex)
        self.loss_tangent = None if loss_tangent is None else float(loss_tangent)
        self.sigma_e = float(sigma_e)
        self.mu_r = mu_r
        self.sigma_m = sigma_m
        self.eps_0 = 8.8541878176e-12
        self.mu_0 =4*np.pi*(1e-7)
        self.c_0 =2.997924580003452e+08
        self.name = name

        if self.getComplexRefractiveIndex(1) == 1:
            self.name = "vacuum"
        
        if (self.epsilon_r > 1e5 and self.mu_r < 1e-5 and \
            self.epsilon_r * self.mu_r == 1) or \
           (self.epsilon_r > 1e5 and self.mu_r >= 0.99 and self.mu_r <= 1.01):
            self.name = "PEC"


    def convertToAbsoptionDepth(self, distance, frequency):
        z = self.getAbsorptionDepth(frequency)
        z = distance/z
        return z

    def convertToLength(self, f, x):
        '''
            f    frequency, type np.array
            x    length (skin depth)
        '''
        z = x*self.getAbsorptionDepth(f)
        return z
    
    def getAbsorptionDepth(self, frequency):
        '''
            x = getAbsorptionDepth(this, frequency) calculates the absorption
            depth from the wave number. The absorption depth is always
            positive regardless the choice of the time-harmonic factor.
            f    frequency, type np.array
        '''
        k = self.getWaveNumber(frequency)
        x = abs(np.real(1j*k))
        return x

    def getComplexPermeability(self, frequency):
        ''' computes the relative complex permeability.
            Input:
                frequency    Nx1 vector (Hz) type np.array

            Note: will fail for frequency = 0
        '''
        mu_r = self.mu_r + self.sigma_m / (1j*2*np.pi*frequency * self.mu_0)
        return mu_r
    
    def getComplexPermittivity(self, frequency):
        ''' computes the relative complex permittivity.
            Input:
                frequency    Nx1 vector (Hz) type np.array

            Note: will fail for frequency = 0
        '''
        frequency = np.asarray(frequency)

        if self.epsilon_r_complex is not None:
            return np.full_like(frequency, self.epsilon_r_complex, dtype=np.complex128)

        if self.loss_tangent is not None:
            return self.epsilon_r * (1 - 1j * self.loss_tangent)

        epsilon_r = self.epsilon_r + self.sigma_e / (1j*2*np.pi*frequency * self.eps_0)
        return epsilon_r
    
    def getComplexRefractiveIndex(self, frequency):
        eps_r = self.getComplexPermittivity(frequency)
        mu_r  = self.getComplexPermeability(frequency)
        ref_idx =  np.sqrt(eps_r*mu_r)
        return ref_idx
    
    def getGroupVelocity(self, frequency):
        '''
            v_g = getGroupVelocity(this, frequency) evalutes the group velocity
            by numerically differentiating the angular frequency with respect to
            the wave number.
        ''' 
        pass
    
    def getIntrinsicImpedance(self, frequency):
        
        eta_0 = np.sqrt(self.mu_0 / self.eps_0)
        eta = eta_0 * \
              np.sqrt( self.getComplexPermeability(frequency) / self.getComplexPermittivity(frequency) )
        return eta
    
    def getPermeability(self):
        ''' returns the object's magnetic permeability
        '''
        return (self.mu_r, self.sigma_m)
    
    def getPermittivity(self):
        ''' returns the object's dielectric permittivity model parameters
        '''
        return (self.epsilon_r, self.sigma_e, self.epsilon_r_complex, self.loss_tangent)
    

    def getPhaseVelocity(self, frequency):
        omega = 2*np.pi*frequency;
        k     = self.getWaveNumber(frequency)
        v_p   = omega/np.real(k);
        return v_p
    
    def getWavelength(self, frequency):
        k = self.getWaveNumber(frequency);
        wavelength = 2*np.pi/np.real(k);
        return wavelength
    
    def getWaveNumber(self, frequency):
        '''
            f    frequency, type np.array
        '''
        permittivity = self.getComplexPermittivity( frequency);
        permeability = self.getComplexPermeability( frequency);
        k = 2*np.pi*(frequency/self.c_0)*np.sqrt((permittivity*permeability));
        return k
    
    def getSkinDepth(self, frequency):
        omega = 2*np.pi*frequency;
        epsilon = self.epsilon_r * self.eps_0
        sigma = self.sigma_e
        mu = self.mu_r * self.mu_0

        skin_depth = np.sqrt( (2/(sigma*mu)) / omega)   *   \
                     np.sqrt( np.sqrt( 1 + ( (epsilon/sigma)*omega )^2   ) + (epsilon/sigma)*omega ) 
        return skin_depth
    