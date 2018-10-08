# This program tests the class RidgeLinearModel vs the
# OLS regression of scikit learn
from utilities import *
from OLSLinearModel import OLSLinearModel
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

x1, x2, y = CreateSampleData(500, 0)
k = 20

# creating instance of OLSLionearModel with order k
my_model = OLSLinearModel(k)
my_model.fit(x1, x2, y)

# creating instance of scikit learn model with order k 
sc_model = LinearRegression()
x = np.concatenate([x1, x2], axis=1)
X = PolynomialFeatures(k).fit_transform(x)
sc_model.fit(X, y)

print("my_model r2score: %f" % my_model.get_R2Score(x1, x2, y))
print("sc_model r2score: %f" % sc_model.score(X, y))
