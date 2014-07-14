import logRegres
from numpy import *

logRegres.deaw()
dataArr,labelMat = logRegres.loadDataSet()

weights = logRegres.stocGradAscent0(array(dataArr), labelMat)
weights = logRegres.gradAscent(dataArr, labelMat)

logRegres.plotBestFit(weights.getA())
