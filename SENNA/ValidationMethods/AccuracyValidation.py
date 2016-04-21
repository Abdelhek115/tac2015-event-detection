from ValidationMethods.ValidationMethod import ValidationMethod

class AccuracyValidation(ValidationMethod):
    """
    Computes the accuracy. Ignores gold labels that are sent in ignoreSetY
    """

    def __init__(self, labelsMapping, ignoreSetY=[]):   
        """
        @param labelsMapping: Mapping label -> int
        @param ignoreSetY: Dict with int values for labels that should be ignored in the computation.
        """     
        super(AccuracyValidation, self).__init__(labelsMapping)
        self.ignoreSetY = ignoreSetY
            
        
        
    def validationMethod(self, evalData, prediction):
        """
        Computes the accurarcy
        """
        setY = evalData.setY
        
        correctCount = 0
        count = 0
        
        for i in xrange(len(setY)):
            if setY[i] in self.ignoreSetY:
                continue
            
            count += 1
            if setY[i] == prediction[i]:
                correctCount += 1
        
        accruacy = float(correctCount) / count if count > 0 else 0
        
        return accruacy, accruacy, accruacy
    
    def savePredictions(self, filePath, datasetName, predictions, goldLabels):
        """
        Stores the prediction at filePath
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
            
    