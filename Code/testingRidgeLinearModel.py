# This program tests the class RidgeLinearModel vs the
# OLS regression of scikit learn
from utilities import *
from OLSLinearModel import OLSLinearModel
from sklearn.linear_model import LinearRegression
import numpy as np

x1, x2, y = CreateSampleData(500, 0)
k = 5

# creating instance of RidgeLionearModel with order k
my_model = OLSLinearModel(k)
my_model.fit(x1, x2, y)

# creating instance of scikit learn model 
sc_model = LinearRegression()
x = np.concatenate([x1,x2], axis=1)
sc_model.fit(x, y)

print("my_model r2score: %f" % my_model.get_R2Score())
print("sc_model r2score: %f" % sc_model.score(x, y))
