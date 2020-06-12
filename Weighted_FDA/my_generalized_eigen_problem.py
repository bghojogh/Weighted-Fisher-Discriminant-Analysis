import numpy as np
from numpy import linalg as LA
from sklearn.metrics.pairwise import pairwise_kernels
from numpy.linalg import inv
# from scipy.linalg import eig
# from scipy.linalg import eigh as eigh_scipy

class My_generalized_eigen_problem:

    def __init__(self, A, B):
        # A Phi = B Phi Lambda --> Phi: eigenvectors, Lambda: eigenvalues
        self.A = A
        self.B = B

    def solve(self):
        Phi_B, Lambda_B = self.eigen_decomposition(matrix=self.B)
        lambda_B = Lambda_B.diagonal()
        a = lambda_B**0.5
        a = np.nan_to_num(a) + 0.0001  #---> for plot --> it worked even for kernel methods in ETH dataset for scale = 0.15
        # a = np.nan_to_num(a) + 0.0000001  #---> for kernel methods in ETH dataset
        # Lambda_B_squareRoot = np.diag(lambda_B**0.5)
        Lambda_B_squareRoot = np.diag(a)
        Phi_B_hat = Phi_B.dot(inv(Lambda_B_squareRoot))
        A_hat = (Phi_B_hat.T).dot(self.A).dot(Phi_B_hat)
        Phi_A, Lambda_A = self.eigen_decomposition(matrix=A_hat)
        Lambda = Lambda_A
        Phi = Phi_B_hat.dot(Phi_A)
        return Phi, Lambda

    def solve_dirty(self):
        C = inv(self.B).dot(self.A)
        # epsilon = 0.00001  # --> to prevent singularity of matrix C
        epsilon = 0.0000001  #--> for ORL faces dataset and ETH dataset
        C = C + epsilon * np.eye(C.shape[0])
        Phi, Lambda = self.eigen_decomposition(matrix=C)
        return Phi, Lambda

    # for MNIST:
    # def solve_dirty(self):
    #     # https://stackoverflow.com/questions/24752393/solve-generalized-eigenvalue-problem-in-numpy
    #     # https://docs.scipy.org/doc/scipy-0.14.0/reference/linalg.html
    #     # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.linalg.eig.html#scipy.linalg.eig
    #     epsilon = 0.0000001  #--> for ORL faces dataset
    #     self.B = self.B + epsilon * np.eye(self.B.shape[0])
    #     eig_val, eig_vec = eigh_scipy(self.A, self.B, eigvals_only=False)
    #     idx = eig_val.argsort()[::-1]  # sort eigenvalues in descending order (largest eigenvalue first)
    #     eig_val = eig_val[idx]
    #     eig_vec = eig_vec[:, idx]
    #     Eigenvectors = eig_vec
    #     eigenvalues = eig_val
    #     eigenvalues = np.asarray(eigenvalues)
    #     Eigenvalues = np.diag(eigenvalues)
    #     return Eigenvectors, Eigenvalues

    def eigen_decomposition(self, matrix):
        eig_val, eig_vec = LA.eigh(matrix)
        idx = eig_val.argsort()[::-1]  # sort eigenvalues in descending order (largest eigenvalue first)
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:, idx]
        Eigenvectors = eig_vec
        eigenvalues = eig_val
        eigenvalues = np.asarray(eigenvalues)
        Eigenvalues = np.diag(eigenvalues)
        return Eigenvectors, Eigenvalues