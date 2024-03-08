# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 15:09:22 2024

@author: Pushkar
"""

import numpy as np

def Ber(z):
    if (np.abs(z) > 1e-9):
        return z / (np.exp(z) -1)
    else:
        return 1.0
def dBer(z):
    if (np.abs(z) > 1e-9):
        return -1*((np.exp(z)*(z-1) + 1)/((np.exp(z)-1)**2))
    else:
        return -0.5
def dBE(z):
    if (np.abs(z) > 1e-9):
        return ((np.exp(z)*(-z+np.exp(z)-1))/((np.exp(z)-1)**2))
    else:
        return 0.5
    
VT = 25e-3
dx = 0.5e-9
Lch = 30e-9
Lg = 10e-9
tinst = 7e-9
tinsb = 5e-9
tsd = 3e-9
Nx = int(Lch/dx) + 1
Nt = int(tinst/dx)
Nb = int(tinsb/dx)
Nsd = int(tsd/dx)
Ng = int(Lg/dx)
Nun = int(0.5*(Lch-Lg)/dx) 
Ny = Nt + Nb + 1
J = np.zeros([Nx*Ny+Nx, Nx*Ny+Nx])
phin = np.zeros(Nx*Ny+Nx)

for j in range(0,Nx):
    for i in range(0,Ny):
        if (i == 0):
            if (j == 0):
                J[i*Nx+j,i*Nx+j] = -2.0
                J[i*Nx+j,i*Nx+j+1] = 1.0
                J[i*Nx+j,(i+1)*Nx+j] = 1.0
            elif (j == Nx-1):
                J[i*Nx+j,i*Nx+j] = -2.0
                J[i*Nx+j,i*Nx+j-1] = 1.0
                J[i*Nx+j,(i+1)*Nx+j] = 1.0
            elif (j >= Nun and j < Nun + Ng):
                J[i*Nx+j,i*Nx+j] = 1.0
            else:
                J[i*Nx+j,i*Nx+j] = -4.0
                J[i*Nx+j,i*Nx+j+1] = 1.0
                J[i*Nx+j,i*Nx+j-1] = 1.0
                J[i*Nx+j,(i+1)*Nx+j] = 2.0
        elif (i == Ny - 1):
            if (j == 0):
                J[i*Nx+j,i*Nx+j] = -2.0
                J[i*Nx+j,i*Nx+j+1] = 1.0
                J[i*Nx+j,(i-1)*Nx+j] = 1.0
            elif (j == Nx-1):
                J[i*Nx+j,i*Nx+j] = -2.0
                J[i*Nx+j,i*Nx+j-1] = 1.0
                J[i*Nx+j,(i-1)*Nx+j] = 1.0
            elif (i >= Nun and j < Nun + Ng):
                J[i*Nx+j,i*Nx+j] = 1.0
            else:
                J[i*Nx+j,i*Nx+j] = -4.0
                J[i*Nx+j,i*Nx+j+1] = 1.0
                J[i*Nx+j,i*Nx+j-1] = 1.0
                J[i*Nx+j,(i-1)*Nx+j] = 2.0
        elif (j == 0):
            if (i >= Nb and i < Nb + Nsd):
                J[i*Nx+j,i*Nx+j] = 1.0
            else:
                J[i*Nx+j,i*Nx+j] = -4.0
                J[i*Nx+j,i*Nx+j+1] = 2.0
                J[i*Nx+j,(i+1)*Nx+j] = 1.0
                J[i*Nx+j,(i-1)*Nx+j] = 1.0
        elif (j == Nx-1):
            if (i >= Nb and i < Nb + Nsd):
                pass
            else:
                J[i*Nx+j,i*Nx+j] = -4.0
                J[i*Nx+j,i*Nx+j-1] = 2.0
                J[i*Nx+j,(i+1)*Nx+j] = 1.0
                J[i*Nx+j,(i-1)*Nx+j] = 1.0
        elif (i == Nb):
            pass
        else:
            J[i*Nx+j,i*Nx+j] = -4.0
            J[i*Nx+j,i*Nx+j+1] = 1.0
            J[i*Nx+j,i*Nx+j-1] = 1.0
            J[i*Nx+j,(i+1)*Nx+j] = 1.0
            J[i*Nx+j,(i-1)*Nx+j] = 1.0


M = Nx*Nb
for i in range(0,Nx):
    if (i == 0):
        J[Nx*Ny +i, Nx*Ny+i] = 1.0
    elif (i == Nx - 1):
        J[Nx*Ny+i, Nx*Ny+i] = 1.0
    else:
        q = (phin[Nx*Nb+i] - phin[Nx*Nb+i-1])/VT
        q1 = (phin[Nx*Nb+i+1] - phin[Nx*Nb+i])/VT
        J[Nx*Ny+i, Nx*Ny+i] = Ber(q) + Ber(q1)*np.exp(q1)
        J[Nx*Ny+i, Nx*Ny+i+1] = -Ber(q1)
        J[Nx*Ny+i, Nx*Ny+i-1] = -Ber(q)*np.exp(q)
        
        J[Nx*Ny+i, Nx*Nb+i-1] = phin[Nx*Ny+i-1]*dBE(q) - phin[Nx*Ny+i]*dBer(q)
        J[Nx*Ny+i, Nx*Nb+i] = -1*phin[Nx*Ny+i-1]*dBer(q)+ phin[Nx*Ny+1]*(dBer(q)-dBE(q1)) + phin[Nx*Ny+i+1]*dBer(q1)
        J[Nx*Ny+i, Nx*Nb+i+1] = phin[Nx*Ny+i]*dBE(q1) - phin[Nx*Ny+i+1]*dBer(q1)
