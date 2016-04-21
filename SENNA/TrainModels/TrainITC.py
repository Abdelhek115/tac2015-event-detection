import theano
import theano.tensor as T
import numpy as np
import time


from SENNA.Layers.MLPITC import MLPITC
from SENNA.TrainModels.TrainModel import TrainModel


"""
Trains a model, using the isolated tag criterion
"""
class TrainITC(TrainModel):    
    batch_size = 35
    L1_reg = 0
    L2_reg = 0
    updateSingleFeatureEmbeddings = False
    
    def __init__(self, embeddingMatrixToUpdate):
        super(TrainITC, self).__init__()
        self.embeddingMatrixToUpdate = embeddingMatrixToUpdate
    
    
    def buildClassifier(self):
        print('%s - build classifier' % (time.strftime("%H:%M:%S", time.localtime(time.time()))))
        rng = np.random.RandomState(1234)     
        self.x = T.imatrix('x')              
        
        self.classifier = MLPITC(rng=rng, input=self.x, n_hidden=self.numHidden, n_out=self.outputLength, 
                                 embeddingsLookups=self.trainData.featureExtractors, embedding_matrix_to_update=self.embeddingMatrixToUpdate)

    
    def buildValidationFunctions(self):
        if not hasattr(self, 'classifier') or self.classifier == None:
            self.buildClassifier()
            
        print('%s - build validation function' % (time.strftime("%H:%M:%S", time.localtime(time.time()))))
        self.predict_labels = theano.function(inputs=[self.x],
                                            outputs=self.classifier.logRegressionLayer.y_pred,                                            
                                            on_unused_input='warn')
    
    def buildTrainingFunctions(self):
        if not hasattr(self, 'classifier') or self.classifier == None:
            self.buildClassifier()
        
        print('%s - build training function' % (time.strftime("%H:%M:%S", time.localtime(time.time()))))
        self.y = T.ivector('y')
        index = T.lscalar()  # index to a [mini]batch
        lr = T.scalar('lr')  # learning rate to use
                

        cost = self.classifier.negative_log_likelihood(self.y) \
                + self.L1_reg * self.classifier.L1 \
                + self.L2_reg * self.classifier.L2_sqr
        
        gparams = []
        for param in self.classifier.params:
            gparam = T.grad(cost, param)
            gparams.append(gparam)
            
        updates = []
        for param, gparam in zip(self.classifier.params, gparams):
            updates.append((param, param - lr * gparam))
            
        if self.updateSingleFeatureEmbeddings: 
            startInput = 0
            startLookup = 0
            
            gradConcatLayer = T.grad(cost, self.classifier.embeddingLayer.output)
            for fExt in self.trainData.featureExtractors:
                updates += fExt.getUpdateRules(self.x, gradConcatLayer, lr, startInput, startLookup) 
                startInput += fExt.getSize()
                startLookup += fExt.getLookupOutputSize()
                
        self.train_x_shared = theano.shared(value=np.asarray(self.trainData.setX, dtype='int32'), name='train_x', borrow=True)
        self.train_y_shared = theano.shared(value=np.asarray(self.trainData.setY, dtype='int32'), name='train_y', borrow=True)
        self.train_model = theano.function(inputs=[index, lr], outputs=[cost],
                                        updates=updates,
                                        givens={
                                            self.x: self.train_x_shared[index * self.batch_size:(index + 1) * self.batch_size],
                                            self.y: self.train_y_shared[index * self.batch_size:(index + 1) * self.batch_size]},
                                        on_unused_input='warn')
        
    
    def startTraining(self, n_epochs): 
        if not hasattr(self, 'train_model') or self.train_model == None:
            self.buildTrainingFunctions()
        print('%s - start training' % (time.strftime("%H:%M:%S", time.localtime(time.time()))))    
        epoch = 0
        done_looping = False
        max_f1 = 0
        max_f1_test = 0
        n_train_batches = self.train_x_shared.get_value(borrow=True).shape[0] / self.batch_size
        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            learning_rate = max((self.start_learning_rate)/(epoch**self.damping_factor), self.min_learning_rate)
            start_time = time.time()
            
        
            acc_cost = 0                
            for minibatch_index in xrange(n_train_batches-1):         
                minibatch_avg_cost = self.train_model(minibatch_index, learning_rate)         
                acc_cost += minibatch_avg_cost[0]
            
            print('%.2f sec for training, learning_rate=%f, avg. cost = %f' % (time.time() - start_time, learning_rate, acc_cost/n_train_batches))   
            
           
            if epoch == 1 or epoch%self.validationFreq == 0:
                devPrediction, prec, rec, f1 = self.evaluateModel(self.devData) 
                precTest, recTest, f1Test = 0,0,0
                
                if self.testData != None:
                    testPrediction, precTest, recTest, f1Test = self.evaluateModel(self.testData) 
                    
                if f1 > max_f1:
                    max_f1 = f1
                    max_f1_test = f1Test
                                
                       
                    self.saveUpdatedEmbeddings(epoch)
                        
                    self.saveResults("dev", epoch, devPrediction, self.devData.setY , f1)   
                    if self.testData != None:
                        self.saveResults("test", epoch, testPrediction, self.testData.setY,  f1Test) 
                        
                    self.saveModel(epoch)   
                                    
                print('epoch %i, precision %f, recall %f, f1 %f (max %f) | test-set: precision %f, recall %f, f1 %f (max %f)' 
                      %  (epoch, prec, rec, f1, max_f1, precTest, recTest, f1Test, max_f1_test))
                
                
                
    def predictLabels(self, dataX, dataSentenceLengths):
        if not hasattr(self, 'predict_labels') or self.predict_labels == None:
            self.buildValidationFunctions()
        
        start_time = time.time()    
        prediction = self.predict_labels(dataX)
        
        print('%.2f sec for validation' % (time.time() - start_time))
        return prediction
    

                    
          
                
    
    
        
        