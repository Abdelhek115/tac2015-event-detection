import numpy
import theano

class Vocabulary:
    """
    Represents a vocabulary with a word2Idx mappings and the corresponding embeddings 
    """
            
    def __init__(self, vocabPath):
        self.word2Idx = {}
        self.embeddings = []
        self.sharedEmbeddings = None
        self.vocabPath = vocabPath
        self.updateSharedEmbeddings = False

        with open(vocabPath, 'r') as fIn:
            idx = 0           
            
            for line in fIn:
                split = line.strip().split(' ')                
                self.embeddings.append(numpy.array([float(num) for num in split[1:]]))
                self.word2Idx[split[0]] = idx
                idx += 1
        
    def getWordIndices(self):
        """
        Returns the mapping of words to their corresponding index in the embeddings array
        """
        return self.word2Idx;
    
    def getEmbeddingSize(self):
        return len(self.embeddings[0])
    
    def getSharedEmbeddings(self):
        """
        Returns the theano shared object for the embeddings matrix
        """
        if self.sharedEmbeddings == None: 
            self.sharedEmbeddings = theano.shared(value=numpy.asarray(self.embeddings, dtype=theano.config.floatX), name=self.vocabPath, borrow=True) 
            

        return self.sharedEmbeddings
    
    def updateSharedEmbeddings(self):
        """
        @return boolean whether the shared embeddings should be updated during training
        """
        return self.updateSharedEmbeddings;
    
    def getVocabName(self):
        """
        @return: Returns the name of the vocab
        """
        filename = self.vocabPath.split('/')[-1]
        return filename[0:-len('.vocab')]
    
    