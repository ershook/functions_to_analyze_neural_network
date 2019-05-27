import numpy as np
import scipy

def pca_eigendecomposition(X, n_pcs):
    
    # 0-center data
    X_centered = X - np.mean(X, axis = 0, keepdims = True) #ie. want columns to have mean zero
    n, m = X_centered.shape
    
    # Compute covariance 
    C = np.dot(X_centered.T, X_centered) / (n-1)
    
    # Eigendecomposition
    ##NOTE: return of np.linalg.eig does not sort by eigenvalues
    eigenvals, eigenvecs = np.linalg.eig(C)
    
    #Sort eigenvals and eigenvecs by decreasing eigenval
    sort_ind = np.argsort(eigenvals)[::-1] #np.argsort returns indices for decreasing values
    eigenvals = eigenvals[sort_ind]
    eigenvecs = eigenvecs[:,sort_ind]
    
    # Project X onto PC space- take top n pcs
    pcs = np.dot(X_centered, eigenvecs)[:,:n_pcs]

    #Compute variance explained
    var_explained_each_pc = eigenvals / np.sum(eigenvals)
    var_explained_by_subspace = np.sum(var_explained_each_pc[:n_pcs])
    print "using %d pcs; explaining %0.1f%% of the variance" % (n_pcs, var_explained_by_subspace * 100.0)
    
    return pcs, eigenvals,var_explained_each_pc, var_explained_by_subspace




def pca_svd(X, n_pcs):
    
    # 0-center data
    X_centered = X - np.mean(X, axis = 0, keepdims = True) #ie. want columns to have mean zero
    n, m = X_centered.shape
    
    # Compute full SVD
    U, s, Vh = np.linalg.svd(X_centered, 
      full_matrices=False)
    
    # Transform X with SVD components
    pcs = np.dot(U[:,:n_pcs], np.diag(s[:n_pcs]))
    var = np.square(s) / (n - 1)
    
    #Compute variance explained
    var_explained_each_pc = s**2.0 / np.sum(s**2.0)
    var_explained_by_subspace = np.sum(var_explained_each_pc[:n_pcs])
    print "using %d pcs; explaining %0.1f%% of the variance" % (n_pcs, var_explained_by_subspace * 100.0)
    
    return pcs, var, var_explained_each_pc, var_explained_by_subspace




# Relationship between singular values and eigen values:
# np.square(Sigma) / (n - 1) == eigen_vals)

# Relationship between eigenvecs and Vh: 
# eigenvecs = Vh.T

# Relationship between svd and eigen projection:
# principle components = svd: U.dot(sigma) = eigen X.dot(eigenvecs)
n, p = 5,3
X = np.random.rand(n, p)
pcs_eigen, eigenvals, var_explained_each_pc, var_explained_by_subspace = pca_eigendecomposition(X,2)
pcs_svd, var, var_explained_each_pc, var_explained_by_subspace = pca_svd(X, 2)

#print pca_eigendecomposition(X,2)
#print pca_svd(X, 2)

# Note SVD has sign ambiguity way of solving this is below. See https://prod-ng.sandia.gov/techlib-noauth/access-control.cgi/2007/076422.pdf

def flip_signs(A, B):
    """
    From https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca
    utility function for resolving the sign ambiguity in SVD
    https://prod-ng.sandia.gov/techlib-noauth/access-control.cgi/2007/076422.pdf
    """
    signs = np.sign(A) * np.sign(B)
    return A, B * signs