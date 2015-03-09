from random import random
from numpy import zeros
from math import tanh
import pylab as pl

class BPNets(object):
    #private properties
    __inputLayerNodeNumber = 0 #the node number of the input layer
    __hideenLayerNodeNumber = 0 #the node number of the hidden layer
    __outputLayerNodeNumber = 0 #the node number of the output layer
    __inputLayerData = [] #the data of the input layer
    __expectedOutputLayerData = [] #the data of the output layer which is expected
    __learningRate = 0 #the learning rate of the network
    __expectedTotalError = 0 #the expected error of the network
    __numberOfCycle = 0 #the max number of the network cycles
    __weightBetweenInputLayerAndHiddenLayer = [] #the weight of the input layer and hidden layer
    __weightBetweenHiddenLayerAndOutputLayer = [] #the weight of the hidden layer and output layer
    __hiddenLayerData = [] #store the data of hidden layer
    
    def __createRandomNumber(self, a, b):
        return ((b - a)*random() + a)
    
    def __dsigmoid(self, a):
        return 1 - a**2
    
    def __makeMatrix(self,I, J, fill=0.0):
        m = []
        for i in range(I):
            m.append([fill]*J)
        return m
    def __initialiseWeight(self):
        weightBetweenInputLayerAndHiddenLayer = []
        for i in range(self.__inputLayerNodeNumber):
            weightBetweenInputLayerAndHiddenLayer.append([0.0]*self.__hideenLayerNodeNumber)
        for i in range(self.__inputLayerNodeNumber):
            for j in range(self.__hideenLayerNodeNumber):
                weightBetweenInputLayerAndHiddenLayer[i][j] = self.__createRandomNumber(0, 1)
                
        weightBetweenHiddenLayerAndOutputLayer = []
        for i in range(self.__hideenLayerNodeNumber):
            weightBetweenHiddenLayerAndOutputLayer.append([0.0]*self.__outputLayerNodeNumber)
        for i in range(self.__hideenLayerNodeNumber):
            for j in range(self.__outputLayerNodeNumber):
                weightBetweenHiddenLayerAndOutputLayer[i][j] = self.__createRandomNumber(0.2, 0.7)
        return weightBetweenInputLayerAndHiddenLayer, weightBetweenHiddenLayerAndOutputLayer
    
    def __initialiseThresholdValue(self):
        thresholdValueOfHiddenLayer = []
        for i in range(self.__hideenLayerNodeNumber):
            thresholdValueOfHiddenLayer.append(0.5)
        
        thresholdValueOfOutputLayer = []
        for i in range(self.__outputLayerNodeNumber):
            thresholdValueOfOutputLayer.append(0.5)
        
        return thresholdValueOfHiddenLayer,thresholdValueOfOutputLayer
    
    #hidden layer uses 'sigmoid' function as its activation function
    def __activationFunctionOfHiddenLayer(self, hiddenLayerInputData):
#         hiddenLayerOutputData = 1/(1 + exp(hiddenLayerInputData))
        hiddenLayerOutputData = tanh(hiddenLayerInputData)
        return hiddenLayerOutputData
    
    #output layer uses 'sigmoid' function as its activation function
    def __activationFunctionOfOutputLayer(self, outputLayerInputData):
        outputLayerOutputData = tanh(outputLayerInputData)
        return outputLayerOutputData
    
    def __dealInputData(self,inputLayerData):
        inputLayerData.append(-1)
        return inputLayerData
    
    #in order to simplify the update process,add threshold value to weight and update the threshold with updating weight
    def __addThresholdValueToWeight(self):
        weight = self.__initialiseWeight()
        thresholdValue = self.__initialiseThresholdValue()
        weightBetweenInputLayerAndHiddenLayer = weight[0]
        weightBetweenHiddenLayerAndOutputLayer = weight[1]
        thresholdValueOfHiddenLayer = thresholdValue[0]
        thresholdValueOfOutputLayer = thresholdValue[1]
        weightBetweenInputLayerAndHiddenLayer.append(thresholdValueOfHiddenLayer)
        weightBetweenHiddenLayerAndOutputLayer.append(thresholdValueOfOutputLayer)
        
#         self.__weightBetweenInputLayerAndHiddenLayer = weightBetweenInputLayerAndHiddenLayer
#         self.__weightBetweenHiddenLayerAndOutputLayer = weightBetweenHiddenLayerAndOutputLayer
        self.__weightBetweenInputLayerAndHiddenLayer,self.__weightBetweenHiddenLayerAndOutputLayer = \
                self.__getInitialWeight()
    #calculate the hidden layer and output layer
    def __forwardCalculation(self, inputLayerData):
        inputLayerData = self.__dealInputData(inputLayerData)
        
        hiddenLayerNodeData = []
        for i in range(self.__hideenLayerNodeNumber):
            hiddenLayerNodeData.append(0)
            
        outputLayerNodeData = []
        for j in range(self.__outputLayerNodeNumber):
            outputLayerNodeData.append(0)
            
        for j in range(self.__hideenLayerNodeNumber):
            hiddenLayerCellData = 0
            for i in range((self.__inputLayerNodeNumber + 1)):
                hiddenLayerCellData = hiddenLayerCellData + inputLayerData[i]*self.__weightBetweenInputLayerAndHiddenLayer[i][j]
            hiddenLayerNodeData[j] = self.__activationFunctionOfHiddenLayer(hiddenLayerCellData)
        
        hiddenLayerNodeData.append(-1)
        for j in range(self.__outputLayerNodeNumber):
            outputLayerCellData = 0
            for i in range((self.__hideenLayerNodeNumber + 1)):
                outputLayerCellData = outputLayerCellData + hiddenLayerNodeData[i]*self.__weightBetweenHiddenLayerAndOutputLayer[i][j]
            outputLayerNodeData[j] = self.__activationFunctionOfOutputLayer(outputLayerCellData)
        
        self.__hiddenLayerData = hiddenLayerNodeData
        return outputLayerNodeData
    
    #calculate the total error of output layer
    def __calculateTotalError(self, outputLayerNodeData, expectedOutputNodeData):
        #calculate the error of the output layer
        errorOfOutputLayerNode = [0.0]*self.__outputLayerNodeNumber
        for i in range(self.__outputLayerNodeNumber):
            errorOfOutputLayerNode[i] = 0.5*(outputLayerNodeData[i] - expectedOutputNodeData[i])**2
        
        #calculate the error of the output layer
        totalErrorOfOutputLayer = 0
        for i in range(self.__outputLayerNodeNumber):
            totalErrorOfOutputLayer = totalErrorOfOutputLayer + errorOfOutputLayerNode[i]
            
        return totalErrorOfOutputLayer
    
    #update weight and threshold value
    def __reverseCalculation(self, outputLayerNodeData, expectedOutputLayerNodeData, learningRate, inputLayerData):
        output_deltas = [0.0] * self.__outputLayerNodeNumber
        for k in range(self.__outputLayerNodeNumber):
            error = expectedOutputLayerNodeData[k]-outputLayerNodeData[k]
            output_deltas[k] = self.__dsigmoid(outputLayerNodeData[k]) * error
        
        hidden_deltas = [0.0] * self.__hideenLayerNodeNumber
        for j in range(self.__hideenLayerNodeNumber):
            error = 0.0
            for k in range(self.__outputLayerNodeNumber):
                error = error + output_deltas[k]*self.__weightBetweenHiddenLayerAndOutputLayer[j][k]
            hidden_deltas[j] = self.__dsigmoid(self.__hiddenLayerData[j]) * error
            
        for j in range(self.__hideenLayerNodeNumber):
            for k in range(self.__outputLayerNodeNumber):
                change = output_deltas[k]*self.__hiddenLayerData[j]
                #Original style of changing the weight
                self.__weightBetweenHiddenLayerAndOutputLayer[j][k] = self.__weightBetweenHiddenLayerAndOutputLayer[j][k] + learningRate*change
#                 #Additional momentum method
#                 self.__weightBetweenHiddenLayerAndOutputLayer[j][k] = \
#                     self.__weightBetweenHiddenLayerAndOutputLayer[j][k] + learningRate*change + self.__mc*self.__co[j][k]
#                 self.__co[j][k] = change

        for i in range(self.__inputLayerNodeNumber):
            for j in range(self.__hideenLayerNodeNumber):
                change = hidden_deltas[j]*inputLayerData[i]
                #Original style of changing the weight
                self.__weightBetweenInputLayerAndHiddenLayer[i][j] = self.__weightBetweenInputLayerAndHiddenLayer[i][j] + learningRate*change
                #Additional momentum method
#                 self.__weightBetweenInputLayerAndHiddenLayer[i][j] = \
#                     self.__weightBetweenInputLayerAndHiddenLayer[i][j] + learningRate*change + self.__mc*self.__ci[i][j]
#                 self.__ci[i][j] = change

#         error = 0.0
#         for k in range(len(expectedOutputLayerNodeData)):
#             error = error + 0.5*(expectedOutputLayerNodeData[k]-outputLayerNodeData[k])**2
#         return error
    def __changeExpectedOutputLayerDataToList(self):
        expectedOutputLayerDataList = []
        for i in self.__expectedOutputLayerData:
            expectedOutputLayerDataList.append(i[0])
        return expectedOutputLayerDataList
    
    def __showErrorWithCycleIndex(self,xLable,yLable):
        pl.plot(xLable,yLable)
        pl.xlabel('CycleIndex')
        pl.ylabel('TotalError')
        pl.title('Additional momentum')
        pl.show()
        
    def runBPNets(self):
        totalErrorCache = [0.0]*self.__numberOfCycle
        xLable = [0.0]*self.__numberOfCycle
        
        for i in range(self.__numberOfCycle):
            outputData= []
            for j,k in enumerate(self.__inputLayerData):
                outputLayerNodeData = self.__forwardCalculation(k)
                self.__reverseCalculation(outputLayerNodeData, self.__expectedOutputLayerData[j], self.__learningRate, k)
                outputData.append(outputLayerNodeData)
            for m in range(len(outputData)):
                totalErrorCache[i] = totalErrorCache[i] + self.__calculateTotalError(outputData[m], self.__expectedOutputLayerData[m])
            xLable[i] = i
            if i > 0:
                if totalErrorCache[i] < totalErrorCache[i-1]:
                    self.__learningRate = self.__learningRate * self.__lrInc
                elif totalErrorCache[i] > 1.04 * totalErrorCache[i-1]:
                    self.__learningRate = self.__learningRate * self.__lrDec
#         self.__showErrorWithCycleIndex(xLable, totalErrorCache)
#                 if i%100 == 0:
#                     print outputLayerData
        
    def testBPNets(self, inputLayerDataForTest, expectedOutputDataForTest):
        totalError = 0
        for i,j in enumerate(inputLayerDataForTest):
            outputLayerNodeDataForOneSample = self.__forwardCalculation(j)
            expectedOutputDataForOneSample = expectedOutputDataForTest[i]
            totalError = totalError + 0.5*(outputLayerNodeDataForOneSample[0] - expectedOutputDataForOneSample[0])**2
            print outputLayerNodeDataForOneSample
        print "TotalError:",totalError
        
    def saveHiddenWeightToTxt(self, txtPath):
        txtFile = open(txtPath,'w')
        
        for i,j in enumerate(self.__weightBetweenInputLayerAndHiddenLayer):
            for m,n in enumerate(j):
                if m == len(j) - 1:
                    txtFile.write(str(n))
                else:
                    txtFile.write(str(n))
                    txtFile.write(' ')
            if i != len(self.__weightBetweenInputLayerAndHiddenLayer) - 1:
                txtFile.write('\n')
        
        txtFile.write('\n')
        for i,j in enumerate(self.__weightBetweenHiddenLayerAndOutputLayer):
            for m,n in enumerate(j):
                if m == len(j) - 1:
                    txtFile.write(str(n))
                else:
                    txtFile.write(str(n))
                    txtFile.write(' ')
            if i != len(self.__weightBetweenHiddenLayerAndOutputLayer) - 1:
                txtFile.write('\n')
        txtFile.close()
        
    def __getInitialWeight(self):
        filePath = r'D:\weight.txt'
        txtFile = open(filePath)
        weightBetweenInputLayerAndHiddenLayer = []
        weightBetweenHiddenLayerAndOutputLayer = []
        j = 0
        for line in txtFile:
            lineArray = line.strip().split(' ')
            for i in range(len(lineArray)):
                cellData = float(lineArray[i])
                lineArray[i] = cellData
            if j < 3:
                weightBetweenInputLayerAndHiddenLayer.append(lineArray)
            else:
                weightBetweenHiddenLayerAndOutputLayer.append(lineArray)
            j = j  + 1
        return weightBetweenInputLayerAndHiddenLayer,weightBetweenHiddenLayerAndOutputLayer
    
    def __init__(self, inputLayerNodeNumber, hiddenLayerNodeNumber, outputLayerNodeNumber,
                 inputLayerData, expectedOutputLayerData, learningRate, 
                 expectedTotalError, numberOfCycle):
        self.__inputLayerNodeNumber = inputLayerNodeNumber
        self.__hideenLayerNodeNumber = hiddenLayerNodeNumber
        self.__outputLayerNodeNumber = outputLayerNodeNumber
        self.__inputLayerData = inputLayerData
        self.__expectedOutputLayerData = expectedOutputLayerData
        self.__learningRate = learningRate
        self.__expectedTotalError = expectedTotalError
        self.__numberOfCycle = numberOfCycle
        self.__addThresholdValueToWeight()
        self.__mc = 0.1
        self.__lrInc = 1.05
        self.__lrDec = 0.7
        self.__co = self.__makeMatrix(self.__hideenLayerNodeNumber, self.__outputLayerNodeNumber)
        self.__ci = self.__makeMatrix(self.__inputLayerNodeNumber, self.__hideenLayerNodeNumber)