from RidgeLinearModel import RidgeLinearModel
import numpy as np
from utilities import *

class OLSLinearModel(RidgeLinearModel):
    def __init__(this, k):
        RidgeLinearModel.__init__(this, 0, k)

    # Computes and returns the covariance matrix of the coefficients of
    # the model function (beta)
    # Only works for OLS
    def get_covariance_matrix(this):
        if this.beta is None:
            print("Error: Model is not fitted.")
            return None
        else:
            if not this.covariance_matrix_updated:
                m = this.x1.shape[0]
                qsigma = (sum(this.y - sum(this.y)/m)/(this.x1.shape[0]
                    - 2 - 1))
                # using pinv instead of inv because inv return negative values
                # on the diagonal
                this.covariance_matrix = np.linalg.pinv(this.X.T.dot(this.X))*qsigma
                this.covariance_matrix_updated = True

            return this.covariance_matrix


    # Computes and returns the variance of the coefficients of the model
    # Only works for OLS
    def get_analytic_variance_of_betas(this):
        if this.beta is None:
            print("Error: Model is not fitted.")
            return None
        else:
            if not this.var_vector_updated:
                var_matrix = this.get_covariance_matrix()
                this.var_vector = np.diagonal(var_matrix)
                this.var_vector_updated = True

            return this.var_vector

    def get_variance_of_betas(this, B=20):
        if this.beta is None:
            print("Error: Model is not fitted.")
            return None
        else:
            if not this.var_vector_updated:
                m = len(this.x1)
                n = SumOneToN(this.k + 1)
                betasamples = np.zeros((n, B))

                for b in range(B):
                    # create bootstrapsample
                    c = np.random.choice(len(this.x1), len(this.x1))
                    s_x1 = this.x1[c]
                    s_x2 = this.x2[c]
                    s_y = this.y[c]
                    # Next line fixes if y is one-dimensional
                    if (len(s_y.shape)) == 1:
                        s_y = np.expand_dims(this.y[c], axis=1)

                    # allocate design matrix
                    s_X = np.ones((m, n))

                    # compute values of design matrix
                    for i in range(m): # vectoriser denne løkka
                        for p in range(this.k):
                            for j in range(SumOneToN(p + 2) - SumOneToN(p + 1)):
                                s_X[i][SumOneToN(p + 1) + j] *= s_x1[i]**(p
                                        + 1 - j)*s_x2[i]**j

                    betasamples[:,b] = np.linalg.pinv(s_X.T.dot(s_X)).dot(s_X.T).dot(s_y)[:, 0]

                betameans = betasamples.sum(axis=1, keepdims=True)/B

                # Compute variance vector
                this.var_vector = np.sum((betasamples - betameans)**2, axis=1)/B
                this.var_vector_updated = True

            return this.var_vector
