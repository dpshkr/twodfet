import numpy as np
from scipy.sparse import coo_array, csr_array
from scipy.sparse.linalg import cg, gmres, spsolve
import scipy.constants as cnsts
import matplotlib.pyplot as plt
import time

class DoubleGateTwoDFET(object):
    def __init__(self,params):
        self.__dx = params['sim']['dx']
        self.__D = params['material']['D']
        self.__kins = params['material']['kins']
        self.__Nx  = round(params['device']['L']/params['sim']['dx']) + 1
        self.__Ny  = 2*round(params['device']['tins']/params['sim']['dx']) + 1
        self.__Nch = round(params['device']['tins']/params['sim']['dx'])
        self.__Nsd = round(params['device']['tsd']/params['sim']['dx'])
        self.__VT = (cnsts.k*params['device']['T'])/(cnsts.e)
        self.__NC = (params['material']['gs']*params['material']['gs']\
            *params['material']['me']*cnsts.m_e*cnsts.k*params['device']['T'])/\
                (cnsts.pi*cnsts.hbar*cnsts.hbar)
        self.__NDOP = self.__NC*np.log(1.0 + np.exp(-self.__D / self.__VT))
        self.__B = np.zeros(self.__Nx * self.__Ny)
        self.phi = np.zeros(self.__Nx * self.__Ny)
        self.__phi_old = np.zeros(self.__Nx * self.__Ny)
        self.n = np.zeros(self.__Nx)
        self.__setup_poisson()

    def __setup_poisson(self):
        Nx = self.__Nx
        Ny = self.__Ny
        data = np.zeros(5*Nx*Ny)
        row = np.zeros(5*Nx*Ny)
        col = np.zeros(5*Nx*Ny)
        count = 0
        # Form a symmetric matrix without any Dirichlet boundary condition
        for i in range(0,Ny):
            for j in range(0,Nx):
                if (i == 0):
                    if (j == 0):
                        data[count] = 1.0
                        row[count] = i*Nx+j
                        col[count] = i*Nx+j
                        count += 1
                        data[count] = -0.5
                        row[count] = i*Nx+j
                        col[count] = i*Nx+j+1
                        count += 1
                        data[count] = -0.5
                        row[count] = i*Nx+j
                        col[count] = (i+1)*Nx+j
                        count += 1
                    elif (j == Nx-1):
                        data[count] = 1.0
                        row[count] = i*Nx+j
                        col[count] = i*Nx+j
                        count += 1
                        data[count] = -0.5
                        row[count] = i*Nx+j
                        col[count] = i*Nx+j-1
                        count += 1
                        data[count] = -0.5
                        row[count] = i*Nx+j
                        col[count] = (i+1)*Nx+j
                        count += 1
                    else:
                        data[count] = 2.0
                        row[count] = i*Nx+j
                        col[count] = i*Nx+j
                        count += 1
                        data[count] = -0.5
                        row[count] = i*Nx+j
                        col[count] = i*Nx+j+1
                        count += 1
                        data[count] = -0.5
                        row[count] = i*Nx+j
                        col[count] = i*Nx+j-1
                        count += 1
                        data[count] = -1.0
                        row[count] = i*Nx+j
                        col[count] = (i+1)*Nx+j
                        count += 1
                elif (i == Ny - 1):
                    if (j == 0):
                        data[count] = 1.0
                        row[count] = i*Nx+j
                        col[count] = i*Nx+j
                        count += 1
                        data[count] = -0.5
                        row[count] = i*Nx+j
                        col[count] = i*Nx+j+1
                        count += 1
                        data[count] = -0.5
                        row[count] = i*Nx+j
                        col[count] = (i-1)*Nx+j
                        count += 1
                    elif (j == Nx-1):
                        data[count] = 1.0
                        row[count] = i*Nx+j
                        col[count] = i*Nx+j
                        count += 1
                        data[count] = -0.5
                        row[count] = i*Nx+j
                        col[count] = i*Nx+j-1
                        count += 1
                        data[count] = -0.5
                        row[count] = i*Nx+j
                        col[count] = (i-1)*Nx+j
                        count += 1
                    else:
                        data[count] = 2.0
                        row[count] = i*Nx+j
                        col[count] = i*Nx+j
                        count += 1
                        data[count] = -0.5
                        row[count] = i*Nx+j
                        col[count] = i*Nx+j+1
                        count += 1
                        data[count] = -0.5
                        row[count] = i*Nx+j
                        col[count] = i*Nx+j-1
                        count += 1
                        data[count] = -1.0
                        row[count] = i*Nx+j
                        col[count] = (i-1)*Nx+j
                        count += 1
                elif (j == 0):
                    data[count] = 2.0
                    row[count] = i*Nx+j
                    col[count] = i*Nx+j
                    count += 1
                    data[count] = -1.0
                    row[count] = i*Nx+j
                    col[count] = i*Nx+j+1
                    count += 1
                    data[count] = -0.5
                    row[count] = i*Nx+j
                    col[count] = (i+1)*Nx+j
                    count += 1
                    data[count] = -0.5
                    row[count] = i*Nx+j
                    col[count] = (i-1)*Nx+j
                    count += 1
                elif (j == Nx - 1):
                    data[count] = 2.0
                    row[count] = i*Nx+j
                    col[count] = i*Nx+j
                    count += 1
                    data[count] = -1.0
                    row[count] = i*Nx+j
                    col[count] = i*Nx+j-1
                    count += 1
                    data[count] = -0.5
                    row[count] = i*Nx+j
                    col[count] = (i+1)*Nx+j
                    count += 1
                    data[count] = -0.5
                    row[count] = i*Nx+j
                    col[count] = (i-1)*Nx+j
                    count += 1
                else:
                    data[count] = 4.0
                    row[count] = i*Nx+j
                    col[count] = i*Nx+j
                    count += 1
                    data[count] = -1.0
                    row[count] = i*Nx+j
                    col[count] = i*Nx+j+1
                    count += 1
                    data[count] = -1.0
                    row[count] = i*Nx+j
                    col[count] = i*Nx+j-1
                    count += 1
                    data[count] = -1.0
                    row[count] = i*Nx+j
                    col[count] = (i+1)*Nx+j
                    count += 1
                    data[count] = -1.0
                    row[count] = i*Nx+j
                    col[count] = (i-1)*Nx+j
                    count += 1
                    
        self.__A = coo_array((data,(row,col)), shape=(Nx*Ny, Nx*Ny)).tocsr()

        # Start applying Dirichlet boundary conditions to LHS (A)
        # Bottom gate
        i = 0 ; j = 0
        self.__A[i*Nx+j,i*Nx+j] = 1.0
        self.__A[i*Nx+j,i*Nx+j+1] = 0.0
        self.__A[i*Nx+j,(i+1)*Nx+j] = 0.0
        self.__A[i*Nx+j+1,i*Nx+j] = 0.0
        self.__A[(i+1)*Nx+j,i*Nx+j] = 0.0
        j = Nx-1
        self.__A[i*Nx+j,i*Nx+j] = 1.0
        self.__A[i*Nx+j,i*Nx+j-1] = 0.0
        self.__A[i*Nx+j,(i+1)*Nx+j] = 0.0
        self.__A[i*Nx+j-1,i*Nx+j] = 0.0
        self.__A[(i+1)*Nx+j,i*Nx+j] = 0.0
        for j in range(1,Nx-1):
            self.__A[i*Nx+j,i*Nx+j] = 1.0
            self.__A[i*Nx+j,i*Nx+j+1] = 0.0
            self.__A[i*Nx+j,i*Nx+j-1] = 0.0
            self.__A[i*Nx+j,(i+1)*Nx+j] = 0.0
            self.__A[i*Nx+j+1,i*Nx+j] = 0.0
            self.__A[i*Nx+j-1,i*Nx+j] = 0.0
            self.__A[(i+1)*Nx+j,i*Nx+j] = 0.0

        # Top gate
        i = Ny-1 ; j = 0
        self.__A[i*Nx+j,i*Nx+j] = 1.0
        self.__A[i*Nx+j,i*Nx+j+1] = 0.0
        self.__A[i*Nx+j,(i-1)*Nx+j] = 0.0
        self.__A[i*Nx+j+1,i*Nx+j] = 0.0
        self.__A[(i-1)*Nx+j,i*Nx+j] = 0.0
        j = Nx-1
        self.__A[i*Nx+j,i*Nx+j] = 1.0
        self.__A[i*Nx+j,i*Nx+j-1] = 0.0
        self.__A[i*Nx+j,(i-1)*Nx+j] = 0.0
        self.__A[i*Nx+j-1,i*Nx+j] = 0.0
        self.__A[(i-1)*Nx+j,i*Nx+j] = 0.0
        for j in range(1,Nx-1):
            self.__A[i*Nx+j,i*Nx+j] = 1.0
            self.__A[i*Nx+j,i*Nx+j+1] = 0.0
            self.__A[i*Nx+j,i*Nx+j-1] = 0.0
            self.__A[i*Nx+j,(i-1)*Nx+j] = 0.0
            self.__A[i*Nx+j+1,i*Nx+j] = 0.0
            self.__A[i*Nx+j-1,i*Nx+j] = 0.0
            self.__A[(i-1)*Nx+j,i*Nx+j] = 0.0

        # Source
        j = 0
        for i in range(self.__Nch, self.__Nch+self.__Nsd+1):
            self.__A[i*Nx+j,i*Nx+j] = 1.0
            self.__A[i*Nx+j,i*Nx+j+1] = 0.0
            self.__A[i*Nx+j,(i+1)*Nx+j] = 0.0
            self.__A[i*Nx+j,(i-1)*Nx+j] = 0.0
            self.__A[i*Nx+j+1,i*Nx+j] = 0.0
            self.__A[(i+1)*Nx+j,i*Nx+j] = 0.0
            self.__A[(i-1)*Nx+j,i*Nx+j] = 0.0

        # Drain
        j = Nx-1
        for i in range(self.__Nch, self.__Nch+self.__Nsd+1):
            self.__A[i*Nx+j,i*Nx+j] = 1.0
            self.__A[i*Nx+j,i*Nx+j-1] = 0.0
            self.__A[i*Nx+j,(i+1)*Nx+j] = 0.0
            self.__A[i*Nx+j,(i-1)*Nx+j] = 0.0
            self.__A[i*Nx+j-1,i*Nx+j] = 0.0
            self.__A[(i+1)*Nx+j,i*Nx+j] = 0.0
            self.__A[(i-1)*Nx+j,i*Nx+j] = 0.0

        '''
        # Useful to check if the resultant matrix is symmetric
        x = self.__A.toarray()
        print((x == x.T).all())
        print(x)
        '''

    def __update_poisson_rhs_dd(self):
        pass

    def __update_poisson_rhs_ballistic(self):
        pass

    def __update_poisson_rhs_equilibrium(self):
        K = (self.__dx)/(self.__kins*cnsts.epsilon_0)
        Nx = self.__Nx
        i  = self.__Nch
        self.n = self.__NC*np.log(1.0 + np.exp((-self.__D+self.__phi_old[i*Nx:i*Nx+Nx])/self.__VT))
        self.__B[i*Nx+1:i*Nx+Nx-1] = K*cnsts.e*(self.__NDOP - self.n[1:Nx-1])


    def __update_poisson_rhs_dirichlet(self,VGT,VGB,VD,VS):
        # Apply Dirichlet Boundary Conditions to RHS (B)
        # Bottom gate
        Nx = self.__Nx ; Ny = self.__Ny
        i = 0 ; j = 0
        self.__B[i*Nx+j] = VGB
        self.__B[(i+1)*Nx+j] -= -0.5*VGB
        j = Nx-1
        self.__B[i*Nx+j] = VGB
        self.__B[(i+1)*Nx+j] -= -0.5*VGB

        for j in range(1,Nx-1):
            self.__B[i*Nx+j] = VGB
            self.__B[(i+1)*Nx+j] -= -1.0*VGB


        # Top gate
        i = Ny-1 ; j = 0
        self.__B[i*Nx+j] = VGT
        self.__B[(i-1)*Nx+j] -= -0.5*VGT
        j = Nx-1
        self.__B[i*Nx+j] = VGT
        self.__B[(i-1)*Nx+j] -= -0.5*VGT
        for j in range(1,Nx-1):
            self.__B[i*Nx+j] = VGT
            self.__B[(i-1)*Nx+j] -= -1.0*VGT

        # Source
        j = 0
        i = self.__Nch
        self.__B[i*Nx+j]  = VS
        self.__B[i*Nx+j+1] -= -1.0*VS
        self.__B[(i-1)*Nx+j] -= -0.5*VS
        i = self.__Nch + self.__Nsd
        self.__B[i*Nx+j]  = VS
        self.__B[i*Nx+j+1] -= -1.0*VS
        self.__B[(i+1)*Nx+j] -= -0.5*VS
        for i in range(self.__Nch+1, self.__Nch+self.__Nsd):
            self.__B[i*Nx+j]  = VS
            self.__B[i*Nx+j+1] -= -1.0*VS

        # Drain
        j = Nx-1
        i = self.__Nch
        self.__B[i*Nx+j]  = VD
        self.__B[i*Nx+j-1] -= -1.0*VD
        self.__B[(i-1)*Nx+j] -= -0.5*VD
        i = self.__Nch + self.__Nsd
        self.__B[i*Nx+j]  = VD
        self.__B[i*Nx+j-1] -= -1.0*VD
        self.__B[(i+1)*Nx+j] -= -0.5*VD
        for i in range(self.__Nch+1, self.__Nch+self.__Nsd):
            self.__B[i*Nx+j]  = VD
            self.__B[i*Nx+j-1] -= -1.0*VD


    def solve_poisson(self, VGT, VGB, VD, VS = 0.0):
        self.__update_poisson_rhs_dirichlet(VGT,VGB,VD,VS)
        start_time = time.time()
        for j in range(0,1):
            self.phi = cg(self.__A,self.__B)
        print("--- %s seconds ---" % (time.time() - start_time))


    def apply_bias_equilibrium(self, VGT, VGB, max_error=1e-4, max_iter=100, alpha=0.1):
        self.__update_poisson_rhs_dirichlet(VGT,VGB,0.0,0.0)
        self.__update_poisson_rhs_equilibrium()
        error = max_error+1.0
        iter_ = 0
        print("Iter no\t\terror")
        print("")
        print("=================================")
        print("")
        while(error > max_error and iter_ < max_iter):
            self.__update_poisson_rhs_equilibrium()
            (self.phi, error_no) = cg(self.__A, self.__B)
            if (error_no != 0):
                raise Exception("Solver not converging")
            error = np.sqrt((((self.phi - self.__phi_old)**2).sum())/(self.__Nx*self.__Ny))
            self.__phi_old = alpha*self.phi + (1.0-alpha)*self.__phi_old
            iter_ += 1
            print(f"{iter_}\t\t{error}")

    def plot(self):
        plt.imshow(self.phi.reshape([self.__Ny,self.__Nx]))
        plt.show()

    def plot_band_diagram(self):
        i = self.__Nch
        Nx = self.__Nx
        plt.plot(self.__D - self.phi[i*Nx:i*Nx+Nx],'-b', linewidth=3)
        plt.show()


params = {'device':{'L':5e-9, 'tins': 5e-9, 'tsd':2e-9,'T':295},
          'sim':{'dx':0.05e-9},
          'material':{'kins':5.0,'gs':2.0,'gv':2.0,'me':0.45,'D':0.1}}
d = DoubleGateTwoDFET(params)
VG = -0.2
d.apply_bias_equilibrium(VG,VG)
d.plot_band_diagram()
