import numpy as np
from scipy.special import gamma

class FEO:
    def __init__(self, temperature, L2_regularized=False, w=0.1):
        """
        Builds a multivariate linear regression model that is trained with free energy optimization

        Initialize: 
        temperature: float, temperature setting
        L2_regularized: include L2_regularization or not, default=False
        w: L2 weight, default=0.1, ignored if L2_regularized=False

        Methods:
        FEO.fit(x, y, fixed_u_star, u_star_mul)
            fit the model with training sample (x, y)
            x: numpy.ndarray, shape = (n, d) where n is number of training points, d is number of features of x
            y: numpy.ndarray, shape = (n, 1) where n is number of training points
            fixed_u_star: whether to compute optimal loss based on temperature, default=False 
            u_star_mul: float, manually set the optimal loss=u_star_mul, ignored if fixed_u_star=False
        
        FEO.predict(x, n_models)
            predict with new input
        """
                
        self.temp = temperature
        self.L2_regularized = L2_regularized
        self.w = w
        self.fitted = False
    
    def fit(self, x, y, fixed_u_star=False, u_star_mul=1.5):
        m = x.shape[0]
        n = x.shape[1]
        self.m = m
        self.n = n
        assert m == len(y), f"length of y ({len(y)}) must match length of x ({m})"
        A = x.T @ x
        if self.L2_regularized:
            A += self.w*np.eye(len(A))
            
        rank = np.linalg.matrix_rank(A)
        self.rank = rank
        if rank < n:
            self._fit_underdetermined(x, y, rank, A, fixed_u_star, u_star_mul)
        else:
            self._fit_determined(x, y, n, A, fixed_u_star, u_star_mul)

    def predict(self, x, n_models=100):
        if not self.fitted:
            print("Model not fitted yet!")
            return
        
        theta_samples = self._sample(n_models)
        self.theta_samples = theta_samples
        y_pred_FEO = x @ theta_samples
        return y_pred_FEO

    def _get_conics(self, n, A, b, c0):
        # center & eigendecomposition of A
        theta_c = (-0.5 * b @ np.linalg.inv(A)).reshape(-1, 1)
        Lambda, Q = np.linalg.eig(A)
        Lambda = np.real_if_close(Lambda, tol=1e-15)

        # solve ellipse equation given some u value 
        u_min = theta_c.T @ A @ theta_c + b @ theta_c + c0
        u_min = u_min.item()
        if u_min < 0:  # numerical issues where u_min=0 is computed as a very small negative value
            u_min = 0

        # calculate scaling factor for circumference with respect to u^((n-1)/2)
        u_ = 1
        axes = Lambda/(theta_c.T @ A @ theta_c - (c0 - (u_min + u_)))
        axes = (np.sqrt(1/axes)).reshape(-1)

        c_ = self._circumference(axes, len(axes))
        factor = c_/u_**((n-1)/2)

        self.A = A
        self.b = b
        self.c0 = c0
        self.u_min = u_min
        self.factor = factor
        self.theta_c = theta_c
        self.Lambda = Lambda
        self.Q = Q

    def _fit_determined(self, x, y, n, A, fixed_u_star, u_star_mul):
        b = -2* y.T @ x
        c0 = y.T @ y
        self._get_conics(n, A, b, c0)

        # calculate the optimal u value, u_star for given temperature
        if fixed_u_star:
            #u_star = np.linalg.norm(self.theta_c) * u_star_mul
            u_star = u_star_mul #self.u_min * u_star_mul
        else:
            u_star = self.temp * self.factor * (n-1)/2
        if u_star < self.u_min + 1e-6: 
            u_star = self.u_min + 1e-6 

        c = self.c0 - u_star
        K = self.Lambda/(self.theta_c.T @ A @ self.theta_c - c)
        K = K.reshape(-1)

        self.c = c
        self.u_star = u_star
        self.K = K
        self.fitted = True
    
    def _fit_underdetermined(self, x, y, rank, A, fixed_u_star, u_star_mul):
        b = -2* y.T @ x
        c0 = y.T @ y
        
        theta_ln = np.linalg.pinv(x) @ y
        Theta = self._get_axes(theta_ln, self._nullspace(x))

        A_reduced = Theta.T @ A @ Theta
        b_reduced = b @ Theta

        self._get_conics(rank, A_reduced, b_reduced, c0)

        if fixed_u_star:
            #u_star = np.linalg.norm(theta_ln) * u_star_mul
            u_star = u_star_mul
        else:
            u_star = self.temp * self.factor * (rank-1)/2
        if u_star < 1e-6: 
            u_star = 1e-6 

        c = self.c0 - u_star
        K = self.Lambda/(self.theta_c.T @ A_reduced @ self.theta_c - c)
        K = K.reshape(-1)

        self.u_min = 0
        self.u_star = u_star
        self.c = c
        self.K = K
        self.Theta = Theta
        self.fitted = True

    def _sample(self, n_models):
        if self.rank < self.n:
            sample_n = self.rank
        else:
            sample_n = self.n

        # samples on a unit sphere
        theta_samples = np.random.normal(0, np.ones(sample_n), size=(n_models, sample_n))
        l1 = np.linalg.norm(theta_samples, axis=1)
        theta_samples = theta_samples/l1.reshape(-1, 1)

        # get tangents of the sample points. There needs to be n-1 tangents for a n-d point
        tangent = np.zeros((len(theta_samples), sample_n-1, sample_n))
        for t_i in range(1, sample_n):
            tangent[:, t_i-1, 0] = theta_samples[:, t_i]
            tangent[:, t_i-1, t_i] = -theta_samples[:, 0]
        tangent_o = tangent.copy()

        # transform from sphere to ellipse
        theta_samples = theta_samples @ np.diag(np.sqrt(1/self.K))
        tangent = tangent @ np.diag(np.sqrt(1/self.K))

        # calculate contraction of local area around sample after transformation
        contract = np.sqrt(np.linalg.det((tangent @ tangent.transpose(0, 2, 1))))
        contract = contract/np.sqrt(np.linalg.det((tangent_o @ tangent_o.transpose(0, 2, 1))))
        contract = np.nan_to_num(contract)
        
        # rejection sampling
        p = contract/np.max(contract)
        choose = np.random.binomial(1, p, size=len(p)).astype(bool)
        theta_samples = theta_samples[choose]

        # rotate according to eigenvectors and shift back to center
        #theta_samples = theta_samples @ self.Q + self.theta_c.reshape(1, -1)
        theta_samples = (self.Q @ theta_samples.T + self.theta_c)
        
        if self.rank < self.n:
            theta_samples = self.Theta @ theta_samples

        return theta_samples

    def _double_factorial(self, n):
        result = 1
        for i in range(n, 0, -2):
            result *= i
        return result

    def _circumference(self, a, n):
        if n%2 == 0:
            D = np.pi/2 * self._double_factorial(n-1) / self._double_factorial(n-2)
        else:
            D = self._double_factorial(n-1) / self._double_factorial(n-2)
        p = np.log(n)/np.log(D)
        H = ((np.prod(a)**p)*(np.sum(a**(-p))/n))**(1/p)
        return 2*np.pi**(n/2)*H/gamma(n/2)
    
    def _nd_cross(self, vectors):
        """
        input: vectors (n x n-1) where n is number of dimension, and n-1 is number of vectors

        return: a (n x 1) vector that is orthogonal to the input vectors
        """
        n = vectors.shape[0]
        assert n == vectors.shape[1] + 1, f"Input vector shape is incorrect. The number of columns ({vectors.shape[1]}) must be number of rows-1 ({n}-1={n-1})"

        cross_vector = np.zeros((n, 1))
        for i in range(n):
            basis = np.zeros((n, 1))
            basis[i] = 1
            cross_vector[i] = np.linalg.det(np.hstack((basis, vectors)))
        return cross_vector

    def _get_axes(self, theta_ln, Null):
        """
        Input: theta_ln is the first axes, Null is the null space, rank is the total number of axes required
        theta_ln shape = (n, 1) where n is number of dimensions/parameters
        Null shape = (n, DOF)

        Output: returns axes of the ellipse, shape = (n, rank), where rank = n - DOF
        the number of columns = rank number of axes, the first of which will be theta_ln, the others will be orthogonal to bothe theta_ln and columns of Null
        """
        n = len(theta_ln)
        #Theta = np.zeros((n, rank))
        #Theta[:, 0] = theta_ln

        # for numerical stability, each vector should be normalized to length 1
        orthogonals = np.hstack((Null, theta_ln))
        orthogonals /= np.linalg.norm(orthogonals, axis=0)
        for i in range(n-Null.shape[1]-1): # number of axes = rank (=n-DOF), number to generate = rank-1 since theta_ln is already 1 axis
            n_orthogonals = orthogonals.shape[1]
            axis = np.zeros((n, 1))
            axis[:n_orthogonals+1] = self._nd_cross(orthogonals[:n_orthogonals+1])
            axis /= np.linalg.norm(axis)
            orthogonals = np.hstack((orthogonals, axis))

        return orthogonals[:, Null.shape[1]:]

    def _nullspace(self, A, atol=1e-13, rtol=0):
        """
        input: A is the underdetermined linear system, shape = (m, n) where m is number of equations, and n is number of variables

        return: null space of A
        """
        A = np.atleast_2d(A)
        u, s, vh = np.linalg.svd(A)
        tol = max(atol, rtol * s[0])
        nnz = (s >= tol).sum()
        ns = vh[nnz:].conj().T
        return ns
    


class EnsembleModel:
    def __init__(self, thetas, method='norm', weighting='recip', alpha=1):
        """
        Builds an ensemble model with thetas

        Initialize: 
        thetas: shape = (n, n_models), where n is number of parameters, and n_models is number of models samples
        method: the method for computing quality score of individual models, choose from 'norm', 'grad', 'var'
        weighting:: the method for weighting individual models using their quality score, choose from 'recip', 'negexp', 'neglog'

        Methods:
        EnsembleModel.predict(x)
        outputs the ensembled model prediction for x 
        """
        assert method in ['norm', 'grad', 'var'], "invalid method, only 'norm', 'grad', 'var' accepted"
        assert weighting in ['recip', 'negexp', 'invsqrt', 'neglog'], "invalid weighting, only 'recip', 'negexp', 'neglog' accepted"

        self.method = method
        self.weighting = weighting
        self.thetas = thetas
        self.alpha = alpha

        if self.method == 'norm':
            q = np.linalg.norm(thetas, axis=0)
            if self.weighting == 'recip':
                self.weights = 1/q**alpha

            elif self.weighting == 'negexp':
                self.weights = np.exp(-q*alpha)
            
            elif self.weighting == 'neglog':
                self.weights = -np.log(q**alpha)

            self.weights /= np.sum(self.weights)
            
    def predict(self, x):
        if self.method == 'norm':
            return x @ self.thetas @ self.weights

        else:
            if self.method == 'grad':
                # compute the magnitude of the gradient at x
                q = np.zeros((x.shape[0], self.thetas.shape[1]))
                for i, sample in enumerate(x):
                    for j, theta in enumerate(self.thetas.T):
                        q[i, j] = np.linalg.norm(sample * theta)

            elif self.method == 'var':
                #raise NotImplementedError
                q = np.zeros((x.shape[0], self.thetas.shape[1]))
                for i, sample in enumerate(x):
                    # generate neighbors of x
                    neighbors = np.random.normal(0, 1e-6, size=(100, x.shape[1])) + sample.reshape(1, -1)
                    neighbors = neighbors @ self.thetas
                    q[i, :] = np.var(neighbors, axis=0)
            
            if self.weighting == 'recip':
                self.weights = np.exp(-np.log(q)*self.alpha) #1/q**self.alpha

            elif self.weighting == 'negexp':
                self.weights = np.exp(-q*self.alpha)
            
            elif self.weighting == 'neglog':
                self.weights = -np.log(q**self.alpha)

            # normalize the weights across all models for each sample in x, i.e rows add up to 1
            self.weights /= np.sum(self.weights, axis=1).reshape(-1,1)

            # predict on x with thetas, apply the individual weights of each model for each sample
            return np.sum((x @ self.thetas) * self.weights, axis=1)