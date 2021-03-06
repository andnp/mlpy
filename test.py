import mlpy
import numpy as np
from numpy import genfromtxt
import os.path

my_data = genfromtxt('/home/andy/Documents/aquarium/max_level.dat', delimiter=',')

X = my_data[:,1:]
Y = my_data[:,0].reshape(X.shape[0], 1)

dat = mlpy.Data(X, Y)
dat.scaleData()

parameters = {
    "hidden": 50,
    "transfer": 0,
    "optimizer": "adadelta"
}

alg = mlpy.ANN(1)
alg.loadParameters(parameters)
alg.loadData(dat)
alg.reset()

if (os.path.isfile("FishTankNet_0")):
    alg.loadModel("FishTankNet")

alg.optimizeBatch(20, 20)
alg.saveModel("FishTankNet")
X = dat.scaleExternal(X)
P = alg.predict(X)

print 1 - (np.sum(np.abs(P - Y)) / P.shape[0])
