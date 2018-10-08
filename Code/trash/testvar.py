from RidgeLinearModel import RidgeLinearModel
from utilities import *

lmb = 0
k = 3

x1, x2, y = CreateSampleData(100, 0)

model = RidgeLinearModel(lmb, k)
model.fit(x1, x2, y)
print(model.get_variance_vector(100))

