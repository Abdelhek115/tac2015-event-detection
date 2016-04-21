from FeatureExtractor import FeatureExtractor
import theano
import theano.tensor as T

class EmbeddingsLookup(FeatureExtractor):
    
    
    def __init__(self, name, embeddingSize=None, sharedEmbeddings=None, update=False):
        self.name = name
        self.embeddingSize = embeddingSize
        self.sharedEmbeddings = sharedEmbeddings
        self.update = update
        
        self.windowSize = 1
        self.featureSize = 1
        
    def __str__( self ):
        return 'Object EmbeddingsLookup(%s, %d)' % (self.name, self.embeddingSize)
        
    def getSize(self):
        return self.featureSize
    
    def getLookupOutputSize(self):
        return self.windowSize*self.embeddingSize
    
    def lookup(self, input_x, start_input, end_input, n_rows):
        assert end_input - start_input == 1
        return self.sharedEmbeddings[input_x[:,start_input]]
        
    
    def getUpdateRules(self, input_x, gradConcatLayer, lr, startInput, startLookup):
        if self.update:
            print "Update embedding %s" % self.name             
            embeddGrad = gradConcatLayer[:, startLookup:startLookup+self.embeddingSize]
            #embeddGrad = T.grad(cost, output_lookup)[:, startLookup+targetWord*self.embeddingSize:startLookup+(targetWord+1)*self.embeddingSize]     
            update_embeddings = T.inc_subtensor(self.sharedEmbeddings[input_x[:, startInput]], -lr*embeddGrad)
        
            return [(self.sharedEmbeddings, update_embeddings)]
        else:
            return [] 