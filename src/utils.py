import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def create_regression(m, variables, order, include_bias=True, return_non_poly=False):
    x = np.random.random((m, variables))*2-1
    poly = PolynomialFeatures(order, include_bias=include_bias)
    if return_non_poly:
        return poly.fit_transform(x), x
    return poly.fit_transform(x)