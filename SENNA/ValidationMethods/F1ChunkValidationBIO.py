from SENNA.IO.Dataset import Dataset
from SENNA.ValidationMethods.F1ChunkValidation import F1ChunkValidation


class F1ChunkValidationBIO(F1ChunkValidation):
    """
    Computes precision, recall and F1 on a chunk level assuming BIO tagging
    """    
    def computePrecision(self, guessed, correct):
        correctCount = 0
        count = 0
        
        idx = 0
        while idx < len(guessed):
            if guessed[idx][0] == 'B': #A new chunk starts
                count += 1
                
                if guessed[idx] == correct[idx]:
                    idx += 1
                    correctlyFound = True
                    
                    while idx < len(guessed) and guessed[idx][0] == 'I': #Scan until it no longer starts with I
                        if guessed[idx] != correct[idx]:
                            correctlyFound = False
                        
                        idx += 1
                    
                    if idx < len(guessed):
                        if correct[idx][0] == 'I': #The chunk in correct was longer
                            correctlyFound = False
                        
                    
                    if correctlyFound:
                        correctCount += 1
                else:
                    idx += 1
            else:  
                idx += 1
        
        precision = 0
        if count > 0:    
            precision = float(correctCount) / count
            
        return precision
    
    
    
    
if __name__ == "__main__":
    labelMapping = {'O':0, 'B-PER':1, 'I-PER':2, 'B-LOC':3, 'I-LOC':4}
    validation = F1ChunkValidationBIO(labelMapping)
    
    setY = [0,0,1,2,2,0,1,0,0,1,2,1,2]
    pred = [0,0,3,4,4,0,1,0,0,1,2,1,2]
    
    evalData = Dataset(None, setY, 10, None)
    print validation.validationMethod(evalData, pred)
    