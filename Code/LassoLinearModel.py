from RidgeLinearModel import RidgeLinearModel
from sklearn import linear_model
from utilities import SumOneToN
import numpy as np

class LassoLinearModel(RidgeLinearModel):
    def fit(this, x, y):
        # store x ands y for later computations
        this.x = x
        this.y = y

        # calculate the dimensions of the design matrix
        m = x.shape[0]
        n = SumOneToN(this.k + 1)

        # allocate design matrix
        this.X = np.ones((m, n))

        # compute values of design matrix
        for i in range(m):
            for p in range(this.k):
                for j in range(SumOneToN(p + 2) - SumOneToN(p + 1)):
                    this.X[i][SumOneToN(p + 1) + j] *= x[i][0]**(p 
                            + 1 - j)*x[i][1]**j

        # compute linear regression coefficients
        lasso = linear_model.Lasso(alpha=this.lmb)
        lasso.fit(x, y)
        this.beta = lasso.coef_
        print(this.beta)

        # stored statistical parameters are no longer valid
        this.set_updated_to_false()

