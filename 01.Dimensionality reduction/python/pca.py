from sklearn.preprocessing import StandardScaler
import numpy as np
import numpy.linalg as lin

# 고유 벡터 축으로 값 변환
def new_coordinates(X,eigenvectors):
    for i in range(eigenvectors.shape[0]):
        if i == 0:
            new = [X.dot(eigenvectors.T[i])]
        else:
            new = np.concatenate((new,[X.dot(eigenvectors.T[i])]),axis=0)
    return new.T


# simple PCA Function
def MYPCA(X, number):
    scaler = StandardScaler()
    x_std = scaler.fit_transform(X)  # scaling
    features = x_std.T
    cov_matrix = np.cov(features)  # 공분산

    eigenvalues = lin.eig(cov_matrix)[0]  # eigenvalue
    eigenvectors = lin.eig(cov_matrix)[1]  # eigenvector

    new_coordinates(x_std, eigenvectors)

    new_coordinate = new_coordinates(x_std, eigenvectors)

    index = eigenvalues.argsort()
    index = list(index)

    for i in range(number):
        if i == 0:
            new = [new_coordinate[:, index.index(i)]]
        else:
            new = np.concatenate(([new_coordinate[:, index.index(i)]], new), axis=0)
    return new.T