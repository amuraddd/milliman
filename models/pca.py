import numpy as np
from sklearn.decomposition import PCA

def get_pca_components(data, num_components=20):
    """
    Run PCA and get a decompistiion of components to visualize explianed variance.
    Inputs:
    - data: dataframe on which to conduct PCA
    - num_components: number of PCA components needed
    Return:
    - pca: model
    - z: projection back to original features
    - z0: components 
    """
    pca = PCA(n_components=num_components)
    z0 = pca.fit_transform(data)
    z0_inverse = pca.inverse_transform(z0)
    z = np.array([z0_inverse])
    print(f"PCA Components Shape: {z0.shape}")
    return pca, z, z0