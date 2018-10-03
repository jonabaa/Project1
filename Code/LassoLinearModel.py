from RidgeLinearModel import RidgeLinearModel
from sklearn import linear_model
from utilities import *
import numpy as np

class LassoLinearModel(RidgeLinearModel):


    # 
    def fit(this, x1, x2, y, ktimes=20):

        # store x ands y for later computations
        this.x1 = x1
        this.x2 = x2
        this.y = y

        # calculate the dimensions of the design matrix
        m = x1.shape[0]
        n = SumOneToN(this.k + 1)

        betasamples = np.zeros((ktimes, n))

        for b in range(ktimes):
            # Calculates ktimes different beta-arrays
            # using a bootstrap method
            c = np.random.choice(len(x1), len(x1))
            x1 = x1[c]
            x2 = x2[c]
            y = y[c]

            # allocate design matrix
            this.X = np.ones((m, n))

            # compute values of design matrix
            for i in range(m): # vectoriser denne løkka
                for p in range(this.k):
                    for j in range(SumOneToN(p + 2) - SumOneToN(p + 1)):
                        this.X[i][SumOneToN(p + 1) + j] *= x1[i]**(p
                                + 1 - j)*x2[i]**j

            # compute linear regression coefficients
            lasso = linear_model.Lasso(alpha=this.lmb)
            lasso.fit(this.X, y)

            betasamples[b,:] = lasso.coef_[:]

        # the beta is now the mean of the betas calculated
        this.beta = betasamples.sum(axis=0) / ktimes
        # Compute variance vector
        this.var_vector = MSE(betasamples, this.beta)

        # stored statistical parameters are no longer valid
        this.set_updated_to_false()






    """
    def fit_old(this, x1, x2, y):
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
        lasso = linear_model.Lasso(alpha=this.lmb)
        lasso.fit(this.X, y)
        this.beta = lasso.coef_

        # stored statistical parameters are no longer valid
        this.set_updated_to_false()
    """
