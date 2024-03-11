import numpy as np
from scipy.linalg import solve, solve_banded
import matplotlib.pyplot as plt
import scipy.constants as cnsts
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

def Br(z):
    if (np.abs(z) < 1e-9):
        return 1.0
    else:
        return z / (np.exp(z) - 1)
Ber = np.vectorize(Br)

class TwoDFET:
    def __init__(self):
        self.Lg = 10e-9
        self.Lch = 30e-9
        self.tinst = 10e-9
        self.tinsb = 10e-9
        self.tsd = 2.5e-9
        self.dx = 0.5e-9
        self.Nx = int(self.Lch/self.dx) + 1
        self.Nb = int(self.tinsb/self.dx)
        self.Nt = int(self.tinst/self.dx)
        self.Nsd = int(self.tsd/self.dx)
        self.Ny = self.Nb + self.Nt + 1
        self.__J = np.zeros([self.Nx*self.Ny,self.Nx*self.Ny])
        self.__f = np.zeros(self.Nx*self.Ny)
        self.phi = np.zeros(self.Nx*self.Ny)
        self.__phi_old = np.zeros(self.Nx*self.Ny)
        self.n = np.zeros(self.Nx)
        self.__C = np.zeros([3,self.Nx])
        self.__D = np.zeros(self.Nx)
        self.__init_jacobi(0.0, 0.0, 0.0)
        

    def __update_carrier_conc(self):
        VT = 25e-3
        Nx = self.Nx
        Ny = self.Ny
        Ng = int(self.Lg / self.dx)
        Nun = int((Nx - Ng)/2)
        Nb = self.Nb
        NC = (cnsts.m_e*cnsts.k*295)/ (np.pi*cnsts.hbar*cnsts.hbar)
        D = 0.2
        ND = NC*np.exp(-D/VT)
        n0 = ND
        u = (self.__phi_old[Nb*Nx+1:Nb*Nx+Nx] - self.__phi_old[Nb*Nx:Nb*Nx+Nx-1])/VT
        self.__C[1,0] = 1.0
        self.__C[1,Nx-1] = 1.0
        self.__C[1,1:Nx-1] = Ber(u[0:Nx-2]) + Ber(u[1:Nx-1])*np.exp(u[1:Nx-1])
        self.__C[0,2:] = -Ber(u[1:Nx-1])
        self.__C[2,0:Nx-2] = -Ber(u[0:Nx-2])*np.exp(u[0:Nx-2])
        self.__D[0] = n0
        self.__D[Nx-1] = n0
        self.n = solve_banded((1,1), self.__C, self.__D)

    def __init_jacobi(self, VBG, VTG, VD, VS=0.0):
        Nx = self.Nx
        Ny = self.Ny
        Ng = int(self.Lg / self.dx)
        Nun = int((Nx - Ng)/2)
        Nsd = int(self.tsd/self.dx)
        Nb = self.Nb
        NC = (cnsts.m_e*cnsts.k*295)/ (np.pi*cnsts.hbar*cnsts.hbar)
        VT = 25e-3
        D = 0.1
        w = (self.dx*cnsts.e)/(5*cnsts.epsilon_0)
        ND = NC*np.exp(-D/VT)
        for j in range(0,Nx):
            for i in range(0,Ny):
                if (i == 0):
                    if (j == 0):
                        self.__f[i*Nx + j] = -2*self.phi[i*Nx+j] + self.phi[i*Nx+j+1] + self.phi[(i+1)*Nx+j]
                        self.__J[i*Nx+j,i*Nx+j] = -2.0
                        self.__J[i*Nx+j, i*Nx+j+1] = 1.0
                        self.__J[i*Nx+j, (i+1)*Nx+j] = 1.0
                    elif (j == Nx - 1):
                        self.__f[i*Nx + j] = -2*self.phi[i*Nx+j] + self.phi[i*Nx+j-1] + self.phi[(i+1)*Nx+j]
                        self.__J[i*Nx+j, i*Nx+j] = -2.0
                        self.__J[i*Nx+j, i*Nx+j-1] = 1.0
                        self.__J[i*Nx+j, (i+1)*Nx+j] = 1.0
                    elif (j >= Nun and j < Nun + Ng):
                        self.__f[i*Nx+j] = self.phi[i*Nx+j] - VBG
                        self.__J[i*Nx + j, i*Nx+j] = 1.0
                    else:
                        self.__f[i*Nx+j] = -4*self.phi[i*Nx+j] + self.phi[i*Nx+j-1] + self.phi[i*Nx+j+1] + 2*self.phi[(i+1)*Nx+j]
                        self.__J[i*Nx+j, i*Nx+j] = -4.0
                        self.__J[i*Nx+j, i*Nx+j-1] = 1.0
                        self.__J[i*Nx+j, i*Nx+j+1] = 1.0
                        self.__J[i*Nx+j, (i+1)*Nx+j] = 2.0
                elif (i == Ny - 1):
                    if (j == 0):
                        self.__f[i*Nx + j] = -2*self.phi[i*Nx+j] + self.phi[i*Nx+j+1] + self.phi[(i-1)*Nx+j]
                        self.__J[i*Nx+j,i*Nx+j] = -2.0
                        self.__J[i*Nx+j, i*Nx+j+1] = 1.0
                        self.__J[i*Nx+j, (i-1)*Nx+j] = 1.0
                    elif (j == Nx-1):
                        self.__f[i*Nx + j] = -2*self.phi[i*Nx+j] + self.phi[i*Nx+j-1] + self.phi[(i-1)*Nx+j]
                        self.__J[i*Nx+j, i*Nx+j] = -2.0
                        self.__J[i*Nx+j, i*Nx+j-1] = 1.0
                        self.__J[i*Nx+j, (i-1)*Nx+j] = 1.0
                    elif (j >= Nun and j < Nun + Ng):
                        self.__f[i*Nx+j] = self.phi[i*Nx+j] - VTG
                        self.__J[i*Nx + j,i*Nx+j] = 1.0
                    else:
                        self.__f[i*Nx+j] = -4*self.phi[i*Nx+j] + self.phi[i*Nx+j-1] + self.phi[i*Nx+j+1] + 2*self.phi[(i-1)*Nx+j]
                        self.__J[i*Nx+j, i*Nx+j] = -4.0
                        self.__J[i*Nx+j, i*Nx+j-1] = 1.0
                        self.__J[i*Nx+j, i*Nx+j+1] = 1.0
                        self.__J[i*Nx+j, (i-1)*Nx+j] = 2.0
                elif(j == 0):
                    if (i >= Nb and i < Nb + Nsd):
                        self.__f[i*Nx+j] = self.phi[i*Nx+j] - VS
                        self.__J[i*Nx+j, i*Nx + j] = 1.0
                    else:
                        self.__f[i*Nx+j] = -4*self.phi[i*Nx+j] + self.phi[(i+1)*Nx+j] + 2*self.phi[i*Nx+j+1] + self.phi[(i-1)*Nx+j]
                        self.__J[i*Nx+j, i*Nx + j] = -4.0
                        self.__J[i*Nx+j, (i+1)*Nx + j] = 1.0
                        self.__J[i*Nx+j, (i-1)*Nx + j] = 1.0
                        self.__J[i*Nx+j, i*Nx+j+1] = 2.0
                elif (j == Nx-1):
                    if (i >= Nb and i < Nb + Nsd):
                        self.__f[i*Nx+j] = self.phi[i*Nx+j] - VD
                        self.__J[i*Nx+j, i*Nx + j] = 1.0
                    else:
                        self.__f[i*Nx+j] = -4*self.phi[i*Nx+j] + self.phi[(i+1)*Nx+j] + 2*self.phi[i*Nx+j-1] + self.phi[(i-1)*Nx+j]
                        self.__J[i*Nx+j, i*Nx + j] = -4.0
                        self.__J[i*Nx+j, (i+1)*Nx + j] = 1.0
                        self.__J[i*Nx+j, (i-1)*Nx + j] = 1.0
                        self.__J[i*Nx+j, i*Nx+j-1] = 2.0
                elif (i == Nb):
                    self.__f[i*Nx+j] = 2*self.phi[i*Nx+j] - self.phi[(i+1)*Nx+j] - self.phi[(i+1)*Nx+j] - \
                                       w*(ND - self.n[j]*np.exp((self.phi[i*Nx+j] - self.__phi_old[i*Nx+j])/VT))
                    self.__J[i*Nx+j, i*Nx+j] = 2.0 + (w/VT)*self.n[j]*np.exp((self.phi[i*Nx+j] - self.__phi_old[i*Nx+j])/VT)
                    self.__J[i*Nx+j, (i+1)*Nx+j] = -1.0
                    self.__J[i*Nx+j, (i-1)*Nx+j] = -1.0
                else:
                    self.__f[i*Nx+j] = -4*self.phi[i*Nx+j] + self.phi[(i+1)*Nx+j] + self.phi[i*Nx+j-1] + self.phi[(i-1)*Nx+j] + self.phi[i*Nx+j+1]
                    self.__J[i*Nx+j, i*Nx + j] = -4.0
                    self.__J[i*Nx+j, (i+1)*Nx + j] = 1.0
                    self.__J[i*Nx+j, (i-1)*Nx + j] = 1.0
                    self.__J[i*Nx+j, i*Nx+j+1] = 1.0
                    self.__J[i*Nx+j, i*Nx+j-1] = 1.0

        self.__J2 = csr_matrix(self.__J)
    def apply_bias(self,VBG,VTG,VD,VS=0.0):
        alpha = 1e-1
        error = 1.0
        while(error>1e-3):
           self.__update_carrier_conc()
           self.__init_jacobi(VBG,VTG,VD,VS)
           dphi = spsolve(self.__J2, -self.__f)
           self.phi += dphi
           self.__phi_old = alpha*self.phi + (1-alpha)*(self.__phi_old)
           error  = np.sqrt((((self.phi-self.__phi_old)**2).sum())/(self.Nx*self.Ny))
           print(error)
            
            
    def get_current(self):
        VT = 25e-3
        mu = 0.01
        k = (self.phi[self.Nx*self.Nb+1:self.Nx*self.Nb+self.Nx] - self.phi[self.Nx*self.Nb:self.Nx*self.Nb+self.Nx-1])/VT
        J = Ber(k)*(self.n[1:] - self.n[0:self.Nx-1]*np.exp(k))
        
        return ((mu*cnsts.k*295/self.dx)*np.mean(J))

N  = 41
VG = [2]
M = len(VG)
J = np.zeros(N)
VD = np.linspace(0,2.0,N)
t = TwoDFET()
#t.apply_bias(1,1,1)
#plt.imshow(t.phi.reshape([t.Ny, t.Nx]))
#plt.show()

for j in range(0,M):
    for i in range(0,N):
        t.apply_bias(VG[j],VG[j],VD[i])
        J[i] = t.get_current()
    plt.plot(np.abs(J),'*-')
plt.show()

