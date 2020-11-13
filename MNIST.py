import numpy
import scipy.special

# 取消科学计数法输出
numpy.set_printoptions(suppress=True)


class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.InputNodes = inputnodes
        self.HiddenNodes = hiddennodes
        self.OutputNodes = outputnodes
        self.LearningRate = learningrate
        self.wih = numpy.random.normal(0.0, pow(self.HiddenNodes, -0.5), (self.HiddenNodes, self.InputNodes))
        self.who = numpy.random.normal(0.0, pow(self.OutputNodes, -0.5), (self.OutputNodes, self.HiddenNodes))
        self.ActivationFunction = lambda x: scipy.special.expit(x)

    def train(self, inputslist, targetslist):
        inputs = numpy.array(inputslist, ndmin=2).T
        target = numpy.array(targetslist, ndmin=2).T 
        hiddenInputs = numpy.dot(self.wih, inputs)
        hiddenOutputs = self.ActivationFunction(hiddenInputs)
        finalInputs = numpy.dot(self.who, hiddenOutputs)
        finalOutputs = self.ActivationFunction(finalInputs)
        outputsErrors = target - finalOutputs
        hiddenErrors = numpy.dot(self.who.T, outputsErrors)
        self.who += self.LearningRate * numpy.dot((outputsErrors * finalOutputs * (1.0 - finalOutputs)),
                                                  numpy.transpose(hiddenOutputs))
        self.wih += self.LearningRate * numpy.dot((hiddenErrors * hiddenOutputs * (1.0 - hiddenOutputs)),
                                                  numpy.transpose(inputs))

    def query(self, inputslist):
        inputs = numpy.array(inputslist, ndmin=2).T
        hiddenInputs = numpy.dot(self.wih, inputs)
        hiddenOutputs = self.ActivationFunction(hiddenInputs)
        finalInputs = numpy.dot(self.who, hiddenOutputs)
        finalOutputs = self.ActivationFunction(finalInputs)

        return finalOutputs


inputNodes = 784
hiddenNodes = 100
outputNodes = 10
learningRate = 0.2

n = NeuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)

trainDataFile = open('data/mnist_train.csv', 'r')
trainDataList = trainDataFile.readlines()
trainDataFile.close()

for loop in range(5):
    for record in trainDataList:
        allValue = record.split(',')
        inputs = numpy.asfarray(allValue[1:]) / 255 * 0.99 + 0.01
        targets = numpy.zeros(outputNodes) + 0.01
        targets[int(allValue[0])] = 0.99
        n.train(inputs, targets)

testDataFile = open('data/mnist_test.csv', 'r')
testDataList = testDataFile.readlines()
testDataFile.close()

score = 0.0
for record in testDataList:
    allValue = record.split(',')
    correctLabel = int(allValue[0])
    inputs = numpy.asfarray(allValue[1:]) / 255 * 0.99 + 0.01
    label = numpy.argmax(n.query(inputs))
    if label == correctLabel:
        score += 1

print(score / len(testDataList))
