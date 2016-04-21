from abc import ABCMeta
from abc import abstractmethod

class ValidationMethod:
    __metaclass__ = ABCMeta
    
    def __init__(self, labelMapping):
               
        self.labelMapping = labelMapping        
        self.reverseLookup = {}        
        
        for key, value in labelMapping.iteritems():
            self.reverseLookup[value] = key
            
        
        self.datasetTokenMapping = {}
    
    @abstractmethod
    def validationMethod(self, evalData, prediction):
        """ 
        Given evalData and prediction, the method computes precision, recall and F1 
        @return: tuple (precision, recall, F1)
        """
        pass
    
    @abstractmethod
    def savePredictions(self, filePath, datasetName, predictions, goldLabels):
        """
        Stores the prediction at filePath
        """
        pass
    
    def setDataset(self, name, tokens):
        """
        Stores internally the tokens for a certain dataset like dev-set or test-set
        """
        if not hasattr(self, 'datasetTokenMapping'):
            self.datasetTokenMapping = {}
            
        self.datasetTokenMapping[name] = tokens