__author__ = 'jerryrjiang'
from numpy import *
randMat = mat(random.rand(4,4))
invRandMat = randMat.I
myEye = randMat*invRandMat
