import numpy as np
from scipy.sparse import coo_array, csr_array
from scipy.sparse.linalg import cg
import matplotlib.pyplot as plt

class DoubleGateTwoDFET(object):
    def __init__(self,params):
        self.__Nx  = round(params['device']['L']/params['sim']['dx']) + 1
        self.__Ny  = 2*round(params['device']['tins']/params['sim']['dx']) + 1
        self.__Nch = round(params['device']['tins']/params['sim']['dx'])
        self.__Nsd = round(params['device']['tsd']/params['sim']['dx'])
        self.__B = np.zeros(self.__Nx * self.__Ny)
        self.phi = np.zeros(self.__Nx * self.__Ny)
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

    def __update_poisson_rhs(self,VGT, VBT, VD):
        pass

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
        (self.phi,error) = cg(self.__A,self.__B)

    def plot(self):
        plt.imshow(self.phi.reshape([self.__Ny,self.__Nx]))
        plt.show()


params = {'device':{'L':10e-9, 'tins': 5e-9, 'tsd':2e-9}, 'sim':{'dx':0.1e-9}}
d = DoubleGateTwoDFET(params)
d.solve_poisson(-2,-2,1.0,-20.0)
d.plot()
