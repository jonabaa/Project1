from utilities import SumOneToN, RSS, MSE, R2Score
import numpy as np
import scipy.stats as st

class RidgeLinearModel:
    covariance_matrix = None # covariance matrix of the model coefficients
    covariance_matrix_updated = False
    beta = None # coefficients of the modelfunction
    var_vector = None
    var_vector_updated = False
    CIbeta = None # confidence interval of betas
    CIbeta_updated = False
    x1 = None # first predictor of sampledata
    x2 = None # second predictor of sampledata
    y = None # responses of sampledata
    y_tilde = None # model predictions for x
    y_tilde_updated = False


    def __init__(this, lmb, k):
        this.lmb = lmb # set lambda of model
        this.k = k # set order of polynomial


    # This function fits the model to the the sample data
    # using Ridge regression
    #
    # @x: array containing predictors
    # @y: array containing responses
    # @k: the degree of the polynomial to be fitted to the sample data
    # @lmb: lambda, determines the emphasize on minimizing the variance
    #       of the model
    #
    def fit(this, x1, x2, y):
        # store x ands y for later computations
        this.x1 = x1
        this.x2 = x2
        this.y = y

        # calculate the dimensions of the design matrix
        m = x1.shape[0]
        n = SumOneToN(this.k + 1)

        # allocate design matrix
        this.X = np.ones((m, n))

        # compute values of design matrix
        for i in range(m): # vectoriser denne løkka
            for p in range(this.k):
                for j in range(SumOneToN(p + 2) - SumOneToN(p + 1)):
                    this.X[i][SumOneToN(p + 1) + j] *= x1[i]**(p
                            + 1 - j)*x2[i]**j

        # compute linear regression coefficients
        this.beta = np.linalg.pinv(this.X.T.dot(this.X) +
                this.lmb*np.identity(n)).dot(this.X.T).dot(y)

        # stored statistical parameters are no longer valid
        this.set_updated_to_false()


    # Predicts and returns the responses of the predictors with
    # the fitted model if the model is fitted
    #
    # @x1: Columnvector containing the first predictor values
    # @x2: Columnvector containing the second predictor values
    #
    def predict(this, x1, x2):
        if this.beta is None:
            print("Error: Model is not fitted.")
            return None
        else:
            # allocate meshgrid filled with constant term
            y = np.ones(x1.shape)*this.beta[0]

            # compute function values
            for p in range(this.k):
                for j in range(SumOneToN(p + 2) - SumOneToN(p + 1)):
                    y += this.beta[SumOneToN(p + 1)
                            + j]*x1**(p+1-j)*x2**j

            return y
            

    # Returns the residuals of the model squared and summed
    def get_RSS(this, x1, x2, y):
        if this.beta is None:
            print("Error: Model is not fitted.")
            return None
        else:
            y_tilde = this.predict(x1, x2)
            return RSS(y, this.y_tilde)


    # Returns the mean squared error of the model 
    # given the sample data (x1, x2, y)
    # 
    # @x1: vector of first predictor
    # @x2: vector of second predictor
    # @y: vector of responses
    #
    def get_MSE(this, x1, x2, y):
        if this.beta is None:
            print("Error: Model is not fitted.")
            return None
        else:
            y_tilde = this.predict(x1, x2)
            return MSE(y, y_tilde)


    # Returns the R2 score of the model
    def get_R2Score(this, x1, x2, y):
        if this.beta is None:
            print("Error: Model is not fitted.")
            return None
        else:
            y_tilde = this.predict(x1, x2)
            return R2Score(y, y_tilde)


    # Returns the confidence interval of the betas
    def get_CI_of_beta(this, percentile=.95):
        if this.beta is None:
            print("Error: Model is not fitted.")
            return None
        else:
            if not this.CIbeta_updated:

                # stdcoeff is the z-score to the two-sided confidence interval
                stdcoeff = st.norm.ppf((1-percentile)/2)
                this.CI_beta = np.zeros((len(this.beta), 2))
                for i in range(len(this.beta)):
                    this.CI_beta[i][0] = this.beta[i] + stdcoeff*np.sqrt(this.var_vector[i])
                    this.CI_beta[i][1] = this.beta[i] - stdcoeff*np.sqrt(this.var_vector[i])

                this.CIbeta_updated = True
                # CI_beta returns a nx2 matrix with each row
                # representing the confidence interval to the corresponding beta
            return this.CI_beta


    def set_updated_to_false(this):
        covariance_matrix_updated = False
        var_vector_updated = False
        y_tilde_updated = False
        CIbeta_updated = False


