from SENNA.IO.Dataset import Dataset
import abc
from SENNA.ValidationMethods.ValidationMethod import ValidationMethod


class F1ChunkValidation(ValidationMethod):
    __metaclass__ = abc.ABCMeta
    """
    Computes precision, recall and F1 on a chunk level.
    """
    
    
    
    
    def validationMethod(self, evalData, prediction):
        """
        Quick fix: Computes the accurarcy
        """
        goldLabels = evalData.setY
        
        mappedPrediction = [self.reverseLookup[label] for label in prediction]
        mappedGoldLabels = [self.reverseLookup[label] for label in goldLabels]
        
        prec = self.computePrecision(mappedPrediction, mappedGoldLabels)
        rec = self.computeRecall(mappedPrediction, mappedGoldLabels)
        
        f1 = 0
        if (rec+prec) > 0:
            f1 = 2.0 * prec * rec / (prec + rec);
        
        return prec, rec, f1
    
    @abc.abstractmethod
    def computePrecision(self, guessed, correct):
        """
        Computes teh precision. Computation depends on the encoding (IOB vs. BIO)
        """
        pass
    
    def computeRecall(self, guessed, correct):
        return self.computePrecision(correct, guessed) #Recall = Precision aber Gold und Guessed vertauscht
    
    
        
    def savePredictions(self, filePath, datasetName, predictions, goldLabels):
        """
        Stores the predictions and gold labels to a file on disk
        """
        if datasetName not in self.datasetTokenMapping:
            print "Dataset %s not found" % datasetName
            return
        
        tokens = self.datasetTokenMapping[datasetName]
        
        idx = 0;
        with open(filePath, 'w') as fOut:
            
            for sentence in tokens:
                for token_label_tuple in sentence:
                    dkproInstanceId = token_label_tuple[1]['DKProTCInstanceID']
                    token = token_label_tuple[1]['Token[0]']
                    label = self.reverseLookup[predictions[idx]]
                    goldLabel = self.reverseLookup[goldLabels[idx]]
                    
                    if goldLabel != token_label_tuple[0]: #Sanity check
                        raise Exception('gold label different than label from dataset')
                    
                    innerLabelGold = 'O'
                    innerLabelPrediction = 'O'
                    
                    fOut.write('%s\t%s\t%s\t%s\t%s\t%s\n' % (dkproInstanceId, token, goldLabel, innerLabelGold, label, innerLabelPrediction))
                    
                    idx+=1
                
                fOut.write('\n')
            
    
    
if __name__ == "__main__":
    labelMapping = {'O':0, 'B-PER':1, 'I-PER':2, 'B-LOC':3, 'I-LOC':4}
    validation = F1ChunkValidationBIO(labelMapping)
    
    setY = [0,0,1,2,2,0,1,0,0,1,2,1,2]
    pred = [0,0,3,4,4,0,1,0,0,1,2,1,2]
    
    evalData = Dataset(None, setY, 10, None)
    print validation.validationMethod(evalData, pred)
    