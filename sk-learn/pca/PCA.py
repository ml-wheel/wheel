from numpy import *
from sklearn.decomposition import PCA

X = array([[-1, -1, 2], [-2, -1, 5], [-3, -2, 6], [1, 1, 4], [2, 1, 7], [3, 2, 3]])
Y = array([[-2, -3, 6], [-4, -8, -2], [-4, -6, 4]])

pca = PCA(n_components=2)
pca.fit(X)
pca.transform(Y)
