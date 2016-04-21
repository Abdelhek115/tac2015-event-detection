import theano
import theano.tensor as T

"""
This class models the lookup layer of the neural network.
Each feature extractor is invoked, to provide the corresponding lookup value (e.g. a word embedding)
"""
class EmbeddingLayer(object):       
    def __init__(self, input, featuresExtractors):        
        self.n_out = 0
        self.updateRules = []
        n_rows = input.shape[0]      
        
        outputBlocks = []        
        start_input = 0
        end_input = 0
        for fEx in featuresExtractors:
            end_input += fEx.getSize()
            self.n_out += fEx.getLookupOutputSize()
            outputBlocks.append(fEx.lookup(input, start_input, end_input, n_rows))            
            start_input = end_input       
        
            
        self.output = T.concatenate(outputBlocks,axis=1)
        
    def getState(self):
        return None
    
    def setState(self, state):
        pass
        
        
        