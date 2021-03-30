import numpy as np


def sample_cov(X):
    # Check the math here https://www.itl.nist.gov/div898/handbook/pmc/section5/pmc541.htm
    # Estimation of covariance matrices: https://en.wikipedia.org/wiki/Covariance_matrix
    N = X.shape[0] # rows of X: number of observations
    D = X.shape[1] # columns of X: number of variables
    mean_col = 1j*np.ones(D) # has to define a complex number for keeping the imaginary part
    for col_indx in range(D):
        mean_col[col_indx] = np.mean(X[:,col_indx])
    Mx = X - mean_col # Zero mean matrix of X
    S = np.dot(Mx.H,Mx) / (N-1) # sample covariance matrix
    return np.conj(S), Mx


def zca_whitening_matrix1(X0):
    """
    Function to compute ZCA whitening matrix (aka Mahalanobis whitening).
    INPUT:  X0: [N x D] matrix.
        Rows: Observations
        Columns: Variables
    ZCAMatrix: [D x D] transformation matrix
    OUTPUT: Y = (X0 -X_mean)W. Its covariance matrix is identity matrix
    """
    N = X0.shape[0]
    # Sample Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / (N-1)
    sigma0 = np.cov(X0, rowvar=False) # [D x D]
    print(sigma0)
    sigma, Mx = sample_cov(X0)
    print(sigma)

    XhX = np.dot(Mx.H, Mx) # (N-1)*sigma should be the same but there is a conjugate difference. Don't know why...
    # Singular Value Decomposition. X = U * np.diag(S) * V
    U,S,Vh = np.linalg.svd(XhX)
        # U: [D x D] eigenvectors of sigma.
        # S: [D x 1] eigenvalues of sigma.
        # V: [D x D] transpose of U
    # Whitening constant: prevents division by zero
    epsilon = 1e-7
    ZCAMatrix = np.sqrt(N-1) * np.dot(Vh.H, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), Vh))  # [M x M]

    Y = np.dot(Mx, ZCAMatrix)
    print((np.cov(Y, rowvar=False)))

    return Y

if __name__ == "__main__":
    # Check the math here https://www.itl.nist.gov/div898/handbook/pmc/section5/pmc541.htm
    X = np.matrix([[4.0, 2.0, 0.6],
                   [4.2, 2.1, 0.59],
                   [3.9, 2.0, 0.58],
                   [4.3, 2.1, 0.62],
                   [4.1, 2.2, 0.63]])
    X1 = np.matrix([[1+1j,0,0],
                   [0,1+2j,2],
                   [0,0,1+3j]])
    X2 = np.matrix([[1,0,0],
                   [0,1,2],
                   [0,0,1]])
    X3 = np.matrix([[5,5],  # https://www.youtube.com/watch?v=cOUTpqlX-Xs
                   [-1,7]])
    X4 = np.matrix([[3,0], # Strang P 373 example of SVD
                   [4,5],
                    [2,3+10j]])
    #S2 = np.cov(X, rowvar=False)  # The set of #row observations, measuring #column variables
    #print(S2)
    Y = zca_whitening_matrix1(X4)

