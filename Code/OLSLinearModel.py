from RidgeLinearModel import RidgeLinearModel

class OLSLinearModel(RidgeLinearModel):
    def __init__(this, k):
        RidgeLinearModel.__init__(this, 0, k)
