#coding:utf-8
__author__ = 'jerryrjiang'
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt

#样本数据
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

#数据预测
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distance = sqDistances**0.5
    sortedDistIndicies = distance.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

#矩阵数据读取
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        if listFromLine[-1]=="didntLike":
            classLabelVector.append(1)
        elif listFromLine[-1]=="smallDoses":
            classLabelVector.append(2)
        elif listFromLine[-1]=="largeDoses":
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector

#数值归一化
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

#测试数据计算
def datingClassTest(oral_normMat, classLabelVector):
    datingDataMat, datingLabels = file2matrix("./datingTestSet2.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    numTestVecs = 100
    errorCount = 0.0

    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], oral_normMat, classLabelVector, 8)
        print classifierResult,datingLabels[i]
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]):errorCount += 1.0
    print "the total error rate is %f" % (errorCount/float(numTestVecs))



#方法执行
#group, labels = createDataSet()
#diffMat, sqDiffMat, sqDistances, distance,sortedDistIndicies,classCount,tmp = classify0([0, 0], group, labels, 3)
#print sqDiffMat, sqDistances,distance,sortedDistIndicies,classCount,tmp
returnMat, classLabelVector = file2matrix("./datingTestSet.txt")
#print returnMat.shape[0], len(classLabelVector)
normMat, ranges, minVals = autoNorm(returnMat)
#print returnMat, classLabelVector, normMat

datingClassTest(normMat, classLabelVector)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(normMat[:, 1], normMat[:, 2], 15.0*array(classLabelVector), 15.0*array(classLabelVector))
plt.show()