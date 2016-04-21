"""
MLP, but with a word embeddings lookup in front
"""

__docformat__ = 'restructedtext en'


import theano.tensor as T


from SoftmaxLayer import SoftmaxLayer
from HiddenLayer import HiddenLayer
from EmbeddingLayer import EmbeddingLayer


class MLPITC(object):
    """Multi-Layer Perceptron Class with isolated tag criterion
    """

    def __init__(self, rng, input, n_hidden, n_out, embeddingsLookups, embedding_matrix_to_update):
        """Initialize the parameters for the multilayer perceptron
        """
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.ft_names = []
        
        #for ft in embeddingsLookups:
        #    self.ft_names.append(ft.getName())
        
        # First a lookup layer to map indices to their corresponding embedding vector
        self.embeddingLayer = EmbeddingLayer(input, embeddingsLookups)
        
      

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayer = HiddenLayer(rng=rng, input=self.embeddingLayer.output,
                                       n_in=self.embeddingLayer.n_out, n_out=n_hidden,
                                       activation=T.tanh)

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = SoftmaxLayer(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out)

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = abs(self.hiddenLayer.W).sum() \
                + abs(self.logRegressionLayer.W).sum()

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (self.hiddenLayer.W ** 2).sum() \
                    + (self.logRegressionLayer.W ** 2).sum()

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params + embedding_matrix_to_update
        
    def getParams(self):
        config = {}
        config['n_hidden'] = self.n_hidden
        config['n_out'] = self.n_out
        #config['ft_names'] = self.ft_names
        config['layer_params'] = (self.embeddingLayer.getState(), self.hiddenLayer.getState(), self.logRegressionLayer.getState())
        return config
    
    def setLayerParams(self, layer_params):
        concatLayerParams, hiddenLayerParams, logRegLayerParams = layer_params
        self.embeddingLayer.setState(concatLayerParams)
        self.hiddenLayer.setState(hiddenLayerParams)   
        self.logRegressionLayer.setState(logRegLayerParams)   
        
