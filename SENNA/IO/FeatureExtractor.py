from abc import ABCMeta
from abc import abstractmethod

class FeatureExtractor:
    __metaclass__ = ABCMeta
    
    
    @abstractmethod
    def getSize(self):
        """Returns the number of input neurons required for this feature extractor"""
        pass
    
    @abstractmethod
    def getLookupOutputSize(self):
        """Returns the size after the lookup layer"""
        pass
    
    
    @abstractmethod
    def lookup(self, input_x, start_input, end_input, n_rows):
        """This method returns for an input (from start_input to end_input) the corresponding lookup value,
        e.g. given some word indices, it returns the word embeddings.
        
        Default behavior: Identity-Function
        return input_x[:,start_input:end_input]
        """
        pass
    
    def getUpdateRules(self, input_x, output_lookup, cost, lr, startInput, startLookup):
        """Returns the parameters, that will be updated through training
        Parameters:
        cost: Cost function (log-likelihood)
        lr: learning rate 
        startInput: startin location for this feature extractor in the X vector
        endInput: ending location for this feature extractor in the X vector"""
        return []
    
    
   
    
    