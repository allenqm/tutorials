#! pip install torch torchvision torchtext
 
# Demonstrate - solving a linear system of equations [Done]
# Demonstrate - calculating cross_entropy [Done]
# Demonstrate - Linear regression [Done]
# Demonstrate - PCA [DONE]
import torch
import sklearn
import pandas as pd
import numpy as np
%matplotlib inline
 
 
# Demonstrate - solving a linear system of equations
# Ax = b
 
a = torch.rand((10,10))
b = torch.rand((10,1))
x = a.inverse() @ b #alternatively, torch.mm(a.inverse(), b)
 
# try a nonsquare matrix a
a = torch.tensor([ [1., 1, 1],
                   [2, 3, 4],
                   [3, 5, 2],
                   [4, 2, 5],
                   [5, 4, 3]])
b = torch.tensor([[-10., -3],
                 [ 12, 14],
                 [ 14, 12],
                 [ 16, 16],
                 [ 18, 16]])
x, _ = torch.lstsq(b, a)
 
# Demonstrate - calculating cross_entropy
 
 
# Least Squares in torch
from sklearn.datasets import make_regression
 
ds = make_regression(n_samples=100, n_features=10)
X = torch.from_numpy(ds[0])
y = torch.from_numpy(ds[1]).unsqueeze(-1)
 
w = torch.lstsq(input=y, A=X)
yhat = a @ w[0][:10]
pd.DataFrame({'yhat':yhat.numpy().ravel(), 'y':y.numpy().ravel()}).plot()
 
 
 
 
 
 
 
 
# Eigen decomposition
'''
Eigen vectors are input vectors into the system described by the matrix A
such that A does not change their direction, only their magnitude.
 
Conceptually, these vectors are akin to the primary vectors that would be
used in a coordinate system.
 
You can gleam the relationship with PCA: PCA is trying to vind the latent
factors that describe most of the variation (trying to find the coordinate system)
'''
def PCA_eig(X,k, center=True, scale=False):
   n,p = X.size()
   ones = torch.ones(n).view([n,1])
   h = ((1/n) * torch.mm(ones, ones.t())) if center  else torch.zeros(n*n).view([n,n])
   H = torch.eye(n) - h
   X_center =  torch.mm(H.double(), X.double())
   covariance = 1/(n-1) * torch.mm(X_center.t(), X_center).view(p,p)
   scaling =  torch.sqrt(1/torch.diag(covariance)).double() if scale else torch.ones(p).double()
   scaled_covariance = torch.mm(torch.diag(scaling).view(p,p), covariance)
   eigenvalues, eigenvectors = torch.eig(scaled_covariance, True)
   components = (eigenvectors[:, :k]).t()
   explained_variance = eigenvalues[:k, 0]
   return { 'X':X, 'k':k, 'components':components,    
            'explained_variance':explained_variance }
 
def PCA_test_eig():
   iris = datasets.load_iris()
   iris_data = torch.from_numpy(iris.data)
   n, p = iris_data.size()
   k = p
   X_reduced = PCA(n_components=k).fit(iris.data)
   X_reduced_eig = PCA_eig(iris_data, k)
   comps_isclose = (torch.allclose(
   torch.abs(torch.from_numpy(X_reduced.components_).double()),
   torch.abs(X_reduced_eig['components'])
   ))
   print('Equal Components: ', comps_isclose)
   vars_isclose = (torch.allclose(
   torch.from_numpy(X_reduced.explained_variance_).double(),
   X_reduced_eig['explained_variance']
   ))
   print('Equal Explained Variance: ', vars_isclose)
   return
PCA_test_eig()
 
 
# Singular value decomposition
 
def PCA_svd(X, k, center=True):
   n = X.size()[0]
   ones = torch.ones(n).view([n,1])
   h = ((1/n) * torch.mm(ones, ones.t())) if center  else torch.zeros(n*n).view([n,n])
   H = torch.eye(n) - h
   X_center =  torch.mm(H.double(), X.double())
   u, s, v = torch.svd(X_center)
   components  = v[:k].t()
   explained_variance = torch.mul(s[:k], s[:k])/(n-1)
   return { 'X':X, 'k':k, 'components':components,    
   'explained_variance':explained_variance }
 
def PCA_test_svd():
   iris = datasets.load_iris()
   iris_data = torch.from_numpy(iris.data)
   n, p = iris_data.size()
   k = p
   X_reduced = PCA(n_components=k).fit(iris.data)
   X_reduced_svd = PCA_svd(iris_data, k)
   comps_isclose = (torch.allclose(
   torch.abs(torch.from_numpy(X_reduced.components_).double()),
   torch.abs(X_reduced_svd['components'])
   ))
   print('Equal Components: ', comps_isclose)
   vars_isclose = (torch.allclose(
   torch.from_numpy(X_reduced.explained_variance_).double(),
   X_reduced_svd['explained_variance']
   ))
   print('Equal Explained Variance: ', vars_isclose)
   return
PCA_test_svd()
# Output
# Equal Components:  True
# Equal Explained Variance:  True
 
X = torch.rand(100,10)
center = False
k=5
 
n = X.size()[0]
ones = torch.ones(n).view([n,1])
h = ((1/n) * torch.mm(ones, ones.t())) if center  else torch.zeros(n*n).view([n,n])
H = torch.eye(n) - h
X_center =  torch.mm(H.double(), X.double())
u, s, v = torch.svd(X_center)
components  = v[:k].t()
explained_variance = torch.mul(s[:k], s[:k])/(n-1)