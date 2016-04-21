"""
Implements a convolution layer and a max over time layer as described by
Collobert et al., NLP almost from scratch, section 3.2.2 sentence approach
"""
import theano.tensor as T
import numpy
import theano

class ConvolutionLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=None):
        """
        @param input: Tensor of type T.tensor3() 
        """
        
       
        self.input = input
        self.rng = rng
        self.n_in = n_in #Input length for a single element
        self.n_out = n_out #Equivalent to the number of filters
        self.activation=activation

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))        
        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            
            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros(n_out, dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b
        
        #Conv. Operations
        self.conv_out = T.dot(input, W)
                
        #Compute max over time
        self.max_over_time = T.max(self.conv_out,axis=1)  + b

     
        self.output = (self.max_over_time if activation is None
                       else activation(self.max_over_time))
     
        # parameters of the model
        self.params = [self.W, self.b]
        
    def getState(self):
        return (self.W.get_value(), self.b.get_value())
    
    def setState(self, state):
        W, b = state
        self.W.set_value(new_value=W, borrow=True)
        self.b.set_value(new_value=b, borrow=True)