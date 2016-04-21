"""
Experiment on the TAC 2015 Event Nugget Detection.
Extracts the actions, BIO Encoding
"""
from SENNA.RunExperiment import runExperiment
from SENNA.ValidationMethods.F1ChunkValidationBIO import F1ChunkValidationBIO



trainDataPath = "tacdata/train-data.txt.gz"
devDataPath = "tacdata/dev-data.txt.gz"
testDataPath = "tacdata/test-data.txt.gz"


configPath = "config/config.txt"

epochs = 20

labelsMapping = {'O':0, 'B-EVENT':1, 'I-EVENT':2}
validation = F1ChunkValidationBIO(labelsMapping)

runExperiment('ITC', configPath, trainDataPath, devDataPath, testDataPath, labelsMapping, epochs, validation)