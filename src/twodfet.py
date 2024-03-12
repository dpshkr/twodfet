import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve
import matplotlib.pyplot as plt
import scipy.constants as cnsts
def Ber(z):
    if (np.abs(z) < 1e-9):
        return 1.0
    else:
        return z / (np.exp(z) - 1)
Lch = 40e-9
tins = 10e-9
Lg = 30e-9
dx = 0.5e-9
VT = 25e-3

Nx = int(Lch/dx)+ 1
Ny = int(tins/dx) + 1
Nun = int(0.5*(Lch - Lg)/dx)
Ng = int(Lg/dx)
J = np.zeros([Nx*Ny, Nx*Ny])
fphi = np.zeros(Nx*Ny)
phi = np.zeros(Nx*Ny)
dphi = np.zeros(Nx*Ny)
dn = np.zeros(Nx)
VS = 0
VD = 0.0
VG = 1.0
Jn = np.zeros([Nx,Nx])
Jnb = np.zeros([3,Nx])
fn = np.zeros(Nx)
NC = (cnsts.m_e*cnsts.k*295)/ (np.pi*cnsts.hbar*cnsts.hbar)
D = 0.1
w = (dx*cnsts.e)/(5*cnsts.epsilon_0)
ND = NC*np.exp(-D/VT)
n = ND*np.ones(Nx)

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
        
Jphi = csr_matrix(J)

for r in range(0,1):
    for j in range(0,Nx):
        for i in range(0,Ny):
            if (j == 0):
                fphi[i*Nx+j] = -(phi[i*Nx+j] - VS)
            elif (j == Nx - 1):
                fphi[i*Nx+j] = -(phi[i*Nx+j] - VD)
            elif (i == 0):
                fphi[i*Nx+j] = -(phi[i*Nx+j] - phi[(i+1)*Nx+j] - w*(ND-n[j]))
            elif (i == Ny -1):
                if (j >= Nun and j < Nun + Ng):
                    fphi[i*Nx+j] = -(phi[i*Nx+j] - VG)
                else:
                    fphi[i*Nx+j] = -(-4*phi[i*Nx+j]+2*phi[(i-1)*Nx+j]+phi[i*Nx+j+1]+phi[i*Nx+j-1])
            else:
                fphi[i*Nx+j] = -(-4*phi[i*Nx+j]+phi[(i-1)*Nx+j]+phi[i*Nx+j+1]+phi[i*Nx+j-1]+phi[(i+1)*Nx+j])

    dphi = spsolve(Jphi, fphi)
    phi += dphi
    for i in range(0,Nx):
        if (i == 0):
            fn[i] = -(n[i] - ND)
            Jn[i,i] = 1.0
        elif (i == Nx-1):
            fn[i] = -(n[i] - ND)
            Jn[i,i] = 1.0
        else:
            k1 = (phi[i] - phi[i-1])/VT
            k2 = (phi[i+1] - phi[i])/VT
            fn[i] = -(Ber(k1)*np.exp(k1)*n[i-1] - (Ber(k1)+Ber(k2)*np.exp(k2))*n[i] + Ber(k2)*n[i+1])
            Jn[i,i] = -Ber(k1)-Ber(k2)*np.exp(k2)
            Jn[i,i+1] = Ber(k2)
            Jn[i,i-1] = Ber(k1)*np.exp(k1)
    Jns = csr_matrix(Jn)
    dn = spsolve(Jns, fn)
    n += dn
    print(np.sqrt(((dphi*dphi).sum())/(Nx*Ny)))
    

        
plt.imshow(phi.reshape([Ny,Nx]))
plt.show()
