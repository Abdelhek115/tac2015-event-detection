import time
import cPickle
import theano
import theano.tensor as T
import os
import numpy as np
import datetime
import random

from abc import ABCMeta
from abc import abstractmethod

class TrainModel:
    __metaclass__ = ABCMeta
    
    def __init__(self):
        self.experimentName = ''
        self.numHidden = 100
        self.start_learning_rate=0.1
        self.min_learning_rate = 0.01
        self.damping_factor = 0.7
        self.validationFreq = 1
        self.rndId = random.randint(0,999)

        self.classifier = None
        self.validationMethod = None   #External method to compute the evaluation score like F1 score     
        self.outputPath = None
        self.savePredictionsMethod = None #External method to store the predictions

        
        
    def getConfig(self):
        configStr = ""
        configStr += 'Experiment-Name: %s\n' % self.experimentName
        configStr += 'Hidden-Size: %i\n' % self.numHidden
        configStr += 'Update embeddings: %s\n' % self.updateSingleFeatureEmbeddings
        configStr += 'Start-Learning-Rate: %f\n' % self.start_learning_rate
        configStr += 'Min-Learning-Rate: %f\n' % self.min_learning_rate

        return configStr
    
    @abstractmethod
    def startTraining(self, n_epochs):
        """ Starts the training of the model for epochs number of rounds"""
        pass
    
    @abstractmethod
    def predictLabels(self, dataX, dataSentenceLengths):
        """ Predicts the classes for the elements in dataX. Returns a flat vector, containing the int number fo the classes"""
        pass
    
    @abstractmethod
    def buildClassifier(self):
        """ Builds the classifier (i.e. the neural network) """
        pass
    
    
    def setData(self, trainData, devData, testData=None):
        self.trainData = trainData
        self.devData = devData        
        self.outputLength = np.max(trainData.setY)+1
        self.testData = testData    
        
            
    def evaluateModel(self, evalData):
        prediction = self.predictLabels(evalData.setX, evalData.sentenceLengths)
        
        prec, rec, f1 = self.validationMethod(evalData, prediction)
        
        
        return prediction, prec, rec, f1
      
        
    
    def saveResults(self, datasetName, epoch, predictions, goldLabels, f1):    
        
        if self.outputPath == None:
            return 
         
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")      
        filePath = os.path.join(self.outputPath, '%s_%s_%s_%d_%d_%.4f.txt' % (timestamp, self.experimentName, datasetName, self.rndId, epoch, f1))
        self.savePredictionsMethod(filePath, datasetName, predictions, goldLabels)   
        
    def saveModel(self, epoch):

        fModel = file(os.path.join(self.modelOutputPath, 'model_%s_%d.obj' % (self.experimentName, epoch)) , 'wb')
        configDict = self.classifier.getParams()
        configDict['config'] = self.getConfig()
        cPickle.dump(configDict, fModel, protocol=cPickle.HIGHEST_PROTOCOL)
        fModel.close()
        
        
    def loadModel(self, modelPath):
        fModel = file(modelPath, 'rb')
        params = cPickle.load(fModel)
        fModel.close()
        
        if self.classifier == None:
            self.buildClassifier()
        
        self.numHidden = params['n_hidden']
        self.classifier.setLayerParams(params['layer_params'])
        
  
    