import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt
import tomllib
import scipy.constants as cnsts
def Br(z):
    if (np.abs(z) < 1e-9):
        return 1.0
    else:
        return z / (np.exp(z) - 1)
    
Ber = np.vectorize(Br)

class SGTransistorDD:
    def __init__(self, parameters='params.toml'):
        with open('params.toml', 'rb') as f:
            self.__params = tomllib.load(f)
        Nx = int((self.__params['device']['Lch'])/(self.__params['simulation']['dx'])) + 1
        Ny = int((self.__params['device']['tins'])/(self.__params['simulation']['dx'])) + 1
        Ng = int((self.__params['device']['Lg'])/(self.__params['simulation']['dx'])) + 1
        Nun = int(0.5*(Nx-Ng))
        self.Nx = Nx
        self.Ny = Ny
        self.phi = np.zeros(Nx*Ny)
        self.n = np.zeros(Nx)
        self.__Jn = np.zeros([3,Nx])
        self.__fn = np.zeros(Nx)
        self.__fphi = np.zeros(Nx*Ny)
        J = np.zeros([Nx*Ny, Nx*Ny])
        for j in range(0,Nx):
            for i in range(0,Ny):
                if (j == 0):
                    J[i*Nx+j, i*Nx+j] = 1.0
                elif (j == Nx - 1):
                    J[i*Nx+j, i*Nx+j] = 1.0
                elif (i == 0):
                    J[i*Nx+j, i*Nx+j] = 1.0
                    J[i*Nx+j, (i+1)*Nx+j] = -1.0
                elif (i == Ny - 1):
                    if (j >= Nun and j < Nun + Ng):
                        J[i*Nx +j, i*Nx + j] = 1.0
                    else:
                        J[i*Nx + j, i*Nx+j] = -4.0
                        J[i*Nx + j, (i-1)*Nx + j] = 2.0
                        J[i*Nx + j, i*Nx+j+1] = 1.0
                        J[i*Nx + j, i*Nx+j-1] = 1.0
                else:
                    J[i*Nx + j, i*Nx+j] = -4.0
                    J[i*Nx + j, (i-1)*Nx + j] = 1.0
                    J[i*Nx+j, (i+1)*Nx+j] = 1.0
                    J[i*Nx + j, i*Nx+j+1] = 1.0
                    J[i*Nx + j, i*Nx+j-1] = 1.0
                
        self.__Jphi = csr_matrix(J)
    
    def apply_bias(self,VG,VD,VS=0.0):
        Nx = self.Nx
        Ny = self.Ny
        Ng = int((self.__params['device']['Lg'])/(self.__params['simulation']['dx'])) + 1
        Nun = int(0.5*(Nx-Ng))
        NC = (self.__params['material']['meff']*cnsts.m_e*cnsts.k*self.__params['device']['T'])/ (np.pi*cnsts.hbar*cnsts.hbar)
        D = self.__params['material']['D']
        w = (self.__params['simulation']['dx']*cnsts.e)/(self.__params['material']['kins']*cnsts.epsilon_0)
        VT = (cnsts.k*self.__params['device']['T'])/(cnsts.e)
        ND = NC*np.exp(-D/VT)
        ephi = 1
        en = 101
        while(ephi > 1e-10 or en > 10):
            for j in range(0,Nx):
                for i in range(0,Ny):
                    if (j == 0):
                        self.__fphi[i*Nx+j] = -(self.phi[i*Nx+j] - VS)
                    elif (j == Nx - 1):
                        self.__fphi[i*Nx+j] = -(self.phi[i*Nx+j] - VD)
                    elif (i == 0):
                        self.__fphi[i*Nx+j] = -(self.phi[i*Nx+j] - self.phi[(i+1)*Nx+j] - w*(ND-self.n[j]))
                    elif (i == Ny -1):
                        if (j >= Nun and j < Nun + Ng):
                            self.__fphi[i*Nx+j] = -(self.phi[i*Nx+j] - VG)
                        else:
                            self.__fphi[i*Nx+j] = -(-4*self.phi[i*Nx+j]+2*self.phi[(i-1)*Nx+j]+self.phi[i*Nx+j+1]+self.phi[i*Nx+j-1])
                    else:
                        self.__fphi[i*Nx+j] = -(-4*self.phi[i*Nx+j]+self.phi[(i-1)*Nx+j]+self.phi[i*Nx+j+1]+self.phi[i*Nx+j-1]+self.phi[(i+1)*Nx+j])

            dphi = spsolve(self.__Jphi, self.__fphi)
            self.phi += self.__params['simulation']['w']*dphi
            
            l = (self.phi[1:Nx] - self.phi[0:Nx-1])/VT
            self.__fn[0] = -(self.n[0]-ND)
            self.__fn[Nx-1] = -(self.n[Nx-1] - ND)
            
            self.__fn[1:Nx-1] = -(Ber(l[0:Nx-2])*np.exp(l[0:Nx-2]))*self.n[0:Nx-2] +\
                (Ber(l[0:Nx-2]) + Ber(l[1:Nx-1])*np.exp(l[1:Nx-1]))*self.n[1:Nx-1] -\
                    Ber(l[1:Nx-1])*self.n[2:Nx]
            
            self.__Jn[1,0] = 1.0 
            self.__Jn[1,Nx-1] = 1.0
            self.__Jn[1,1:Nx-1] = -Ber(l[0:Nx-2]) - Ber(l[1:Nx-1])*np.exp(l[1:Nx-1])
            self.__Jn[0,2:Nx] = Ber(l[1:Nx-1])
            self.__Jn[2,0:Nx-2] = Ber(l[0:Nx-2])*np.exp(l[0:Nx-2])
            
            dn = solve_banded((1,1), self.__Jn, self.__fn)
            self.n += dn
            
            ephi = np.sqrt(((self.__fphi*self.__fphi).sum())/(Nx*Ny))
            en = np.sqrt(((self.__fn*self.__fn).sum())/(Nx))
            print(ephi,en)
        
        
    def get_current(self):
        VT = (cnsts.k*self.__params['device']['T'])/(cnsts.e)
        s = ((cnsts.k*self.__params['device']['T']*self.__params['material']['mu']) / (self.__params['simulation']['dx']))
        s = s*self.__params['device']['W']
        Nx = self.Nx
        k = (self.phi[1:Nx] - self.phi[0:Nx-1])/VT
        J = s*(Ber(k[0:Nx-1])*(self.n[1:Nx] - self.n[0:Nx-1]*np.exp(k[0:Nx-1])))
        return np.mean(J)

s = SGTransistorDD()
VG = 2.0
VD = np.linspace(0,1.0,11)
J = np.zeros(11)

for i in range(0,11):
    s.apply_bias(VG, VD[i])
    J[i] = s.get_current()
    print(f"============{i}=====================")
plt.plot(VD, J)
