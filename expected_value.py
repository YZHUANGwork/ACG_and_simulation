import numpy as np



def pmf(scores, axis=-1):
    unnormalized_density = np.exp(scores)
    Z = np.sum(unnormalized_density, axis=axis, keepdims=True)
    return unnormalized_density / Z

def expected_value(X, Y, F):
    d = X.shape[-1]

    dot_products = (X @ Y.T) / np.sqrt(d)
    probability = pmf(dot_products, axis=-1)
    F_smeared = probability @ F
    return F_smeared, probability