from BPNets import BPNets
import time


def timespan(func):
    def _timespan():
        start=time.clock()
        func()
        print 'cost %s'%(time.clock() - start),'s'
    return _timespan

@timespan
def testBPNets():
    inputLayerNodesNubmber = 2
    hiddentLayerNodesNumber = 3
    outputLayerNodesNumber = 1
    inputLayerData = [[0,0],[0,1],[1,0],[1,1]]
    expectedOutputLayerData = [[0], [1], [1], [0]]
    learningRate = 0.5
    iterationNumber = 1500
    inputLayerDataForTest = [[0,1],[0,0],[1,0],[1,1]]
    expectedOutputLayerDataForTest = [[1], [0], [1], [0]]
    dateAndTime = time.strftime('%m-%d-%H-%M-%S',time.localtime(time.time()))+'.txt'
    weightTxtFileOfInputLayerAndHiddenLayer = r'D:\%s'%dateAndTime
    bpNets = BPNets(inputLayerNodesNubmber,hiddentLayerNodesNumber,outputLayerNodesNumber,inputLayerData,
                    expectedOutputLayerData,learningRate,0.001,iterationNumber)
    bpNets.runBPNets()
    bpNets.testBPNets(inputLayerDataForTest, expectedOutputLayerDataForTest)
    bpNets.saveHiddenWeightToTxt(weightTxtFileOfInputLayerAndHiddenLayer)
    
if __name__ == '__main__':
    testBPNets()
    pass