from SENNA.IO.FeatureStore import FeatureStore
from SENNA.TrainModels.TrainITC import TrainITC
import ConfigParser
import numpy as np
from libs.stanford_corenlp_pywrapper import CoreNLP




coreNlpPath = "corenlp/*"
inputPath = "input.txt"
outputPath = "output.txt"
configPath = "config/config.txt"
featureFile = inputPath+".features"
modelPath = 'models/model_eventnuggets.obj'

# :: Convert input to a .features file ::

proc = CoreNLP("pos", corenlp_jars=[coreNlpPath])
fIn = open(inputPath)
text = fIn.read()
res = proc.parse_doc(text)

fOut = open(featureFile, 'w')

def getCasing(word):  
    if word.isdigit(): #Is a digit
        return 'numeric'
    if word.islower(): #All lower case
        return 'allLower'
    elif word.isupper(): #All upper case
        return 'allUpper'
    elif word[0].isupper(): #is a title, initial char upper, then all lower
        return 'initialUpper'
    
    return 'other'
    
def getFeatures(sent, position):    
    DEFAULT = "PADDING"
    features = {}
    for offset in xrange(-3,3+1):
        for fieldName, featureName in {'tokens':'Token', 'lemmas':'Lemma', 'pos':'POS'}.iteritems():
            features[featureName+"["+str(offset)+"]"] = sent[fieldName][position+offset].strip() if (position+offset) >= 0 and (position+offset) < len(sent[fieldName]) else DEFAULT
        
        features["Case["+str(offset)+"]"] = getCasing(sent['tokens'][position+offset]) if (position+offset) >= 0 and (position+offset) < len(sent[fieldName]) else DEFAULT
        
        
    return features

for sentence in res['sentences']:
    for position in xrange(len(sentence['tokens'])):
        features = getFeatures(sentence, position)
        featureString = []            
        for key in sorted(features.keys()):
            featureString.append("%s=%s" % (key, features[key]))
                
        featureString = "\t".join(featureString)
        label = "O"
        fOut.write("%s\t%s\n" % (label, featureString))
        
    fOut.write("\n")

fOut.close()  
    
    



# :: Run the existent model ::

#featureFile = 'tacdata/dev-data.txt'

labelsMapping = {'O':0, 'B-EVENT':1, 'I-EVENT':2}
inverseLabelsMapping = {v: k for k, v in labelsMapping.items()}

config = ConfigParser.ConfigParser()
config.read(configPath)
fixedFeatureNames = set([feature.strip() for feature in config.get('Main', 'featureSet').split(',') if len(feature.strip()) > 0])


featureStore = FeatureStore(config, fixedFeatureNames, {})
featureStore = FeatureStore(config, fixedFeatureNames, {})
featureStore.minSentenceLength =  1
allFeatureNames = fixedFeatureNames
featureStore.labelTransform = None
featureStore.initVocabs()


# Load the embeddings
print "Load embeddings"
featureStore.loadVocabs()


devData  = featureStore.createMatrices(featureFile, labelsMapping)

model = TrainITC([])
model.setData(devData, None, None)
model.numHidden = int(config.get('Main', 'numHidden'))
model.loadModel(modelPath)


predictions = model.predictLabels(devData.setX, devData.sentenceLengths)

#Store the predictions
tokens = featureStore.readFeatures(featureFile, ['Token[0]'])
fOut = open(outputPath, 'w')

predictionIdx = 0
for sentence in tokens:
    for token in sentence:        
        word = token[1]['Token[0]']
        pred = inverseLabelsMapping[predictions[predictionIdx]]
        
        predictionIdx += 1
        
        fOut.write("%s\t%s\n" % (word, pred))
    fOut.write("\n")
fOut.close()
