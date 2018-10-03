from RidgeLinearModel import RidgeLinearModel
import numpy as np

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
    def get_variance_of_coefficients(this):
        if this.beta is None:
            print("Error: Model is not fitted.")
            return None
        else:
            if not this.var_vector_updated:
                var_matrix = this.get_covariance_matrix()
                this.var_vector = np.diagonal(var_matrix)
                this.var_vector_updated = True

            return this.var_vector
