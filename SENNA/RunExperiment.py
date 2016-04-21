import numpy
import ConfigParser
import os
import gc
import sys
import gzip

from IO.FeatureStore import FeatureStore
from TrainModels.TrainITC import TrainITC
import theano.tensor as T


  
def openFile(filename, mode):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)
    
def runExperiment(model, configPath, trainDataPath, devDataPath, testDataPath, labelsMapping, epochs=50, validation=None, labelTransform=None):
 
    print "Load config"
    config = ConfigParser.ConfigParser()
    config.read(configPath)
    
    
    
    if config.has_option('Main', 'featureSet'):
        fixedFeatureNames = set([feature.strip() for feature in config.get('Main', 'featureSet').split(',') if len(feature.strip()) > 0])
    else:
        print "Load all feature names from train file"    
        allFeatureNames = []
        with openFile(trainDataPath, 'r') as fIn:
            line = fIn.readline()
            splits = line.split('\t')
            for i in xrange(1, len(splits)):
                value = splits[i].strip()
                if value == '__BOS__' or value == '__EOS__':
                    continue
    
                valueSplit = value.split('=',1)   
                
                featureName = valueSplit[0]  
                for rule in FeatureStore.replaceRules:
                    featureName = featureName.replace(rule[0], rule[1])      
                
                allFeatureNames.append(featureName)       
                          
            
        fixedFeatureNames = set(allFeatureNames) - set(config.get('Main', 'ignoreFeature').split(','))
    
    convFeatureNames = {}
    
        
    
    featureStore = FeatureStore(config, fixedFeatureNames, convFeatureNames)
    featureStore.minSentenceLength = 2 if model.lower() == "stc" else 1
    
    allFeatureNames = fixedFeatureNames.union(convFeatureNames.keys())
    
    featureStore.labelsToExcludeTrain = config.get('Main', 'labelsToExcludeTrain').split(',') if config.has_option('Main', 'labelsToExcludeTrain') else []    
    featureStore.labelsToExcludeDev = config.get('Main', 'labelsToExcludeDev').split(',') if config.has_option('Main', 'labelsToExcludeDev') else [] 
    featureStore.labelsToExcludeTest = config.get('Main', 'labelsToExcludeTest').split(',') if config.has_option('Main', 'labelsToExcludeTest') else []
        
    featureStore.labelTransform = labelTransform
    
    print "All Feature Names: %s" % sorted(allFeatureNames)
    print "FixedFeatures: %s" % sorted(fixedFeatureNames)
    print "ConvolutionalFeatures: %s" % convFeatureNames
    
    
    
    print "Init vocabs"
    featureStore.initVocabs()
    
    print "Create default vocabs"
    featureStore.createDefaultVocabs(trainDataPath, fixedFeatureNames, convFeatureNames)
    
    
    
    # Load the embeddings
    print "Load embeddings"
    featureStore.loadVocabs()
    
    #Set the embeddings we want to update to true
    if config.has_option('Main', 'updateVocabulariesForFeatures'):
        updateFeatures = config.get('Main', 'updateVocabulariesForFeatures').split(',')
        for feature in updateFeatures:
            feature = feature.strip()
            if len(feature) > 0:
                featureStore.getVocabForFeature(feature).updateSharedEmbeddings = True
    
    
    print "Feature mapping: %s" % featureStore.featureMapping
    
    
    
    #Labels mapping to integers
   
    
    print labelsMapping
    
    # Create the matrices for training (X,Y and lookup matrices)
    print "Create matrices for the training"
    
    
    trainData = featureStore.createMatrices(trainDataPath, labelsMapping, featureStore.labelsToExcludeTrain)
    
    
    print "Unknowns in Train Data (%s):" % str(trainData.setX.shape)
    for featureName in sorted(featureStore.lookupCount):
        if featureName in featureStore.lookupUnkownCount:
            print "%s: %.2f%%" % (featureName, float(featureStore.lookupUnkownCount[featureName])/featureStore.lookupCount[featureName]*100)
    
    
    
    devData  = featureStore.createMatrices(devDataPath, labelsMapping, featureStore.labelsToExcludeDev)
    print "Unknowns in Dev Data (%s):" % str(devData.setX.shape)
    for featureName in sorted(featureStore.lookupCount):
        if featureName in featureStore.lookupUnkownCount:
            print "%s: %.2f%%" % (featureName, float(featureStore.lookupUnkownCount[featureName])/featureStore.lookupCount[featureName]*100)
    
    testData  = featureStore.createMatrices(testDataPath, labelsMapping, featureStore.labelsToExcludeTest) if testDataPath != None else None
    
    if testData != None:
        print "Unknowns in Test Data:"
        for featureName in sorted(featureStore.lookupCount):
            if featureName in featureStore.lookupUnkownCount:
                print "%s: %.2f%%" % (featureName, float(featureStore.lookupUnkownCount[featureName])/featureStore.lookupCount[featureName]*100)
    
    #Valdation method
    
    validation.setDataset("dev", featureStore.readFeatures(devDataPath, ['Token[0]', 'DKProTCInstanceID'], featureStore.labelsToExcludeDev))
    
    if testDataPath != None:
        validation.setDataset("test", featureStore.readFeatures(testDataPath, ['Token[0]', 'DKProTCInstanceID'], featureStore.labelsToExcludeTest))
    
    embeddingMatrixToUpdate = []
    for vocab in featureStore.vocabs.itervalues():
        if vocab.updateSharedEmbeddings:
            print "Update embeddings: "+vocab.vocabPath
            embeddingMatrixToUpdate.append(vocab.sharedEmbeddings)
    
    #Train the network
    if model.lower() == 'itc':        
        model = TrainITC(embeddingMatrixToUpdate)
    elif model.lower() == 'stc':    
        raise ValueError('STC model not included - please contact reimers@ukp.informatik.tu-darmstadt.de')
    else:
        raise ValueError('Unknown model value: '+model)
    
    model.config = config
    model.experimentName = config.get('Main', 'experimentName') if config.has_option('Main', 'experimentName') else ''    
    model.numHidden = int(config.get('Main', 'numHidden'))
    model.setData(trainData, devData, testData)
    model.validationMethod = validation.validationMethod
    model.savePredictionsMethod = validation.savePredictions
    model.updateSingleFeatureEmbeddings = (config.has_option('Main', 'updateSingleFeatureEmbeddings'))
    model.saveUpdatedEmbeddings = featureStore.saveUpdatedEmbeddings
    model.outputPath = config.get('Main', 'outputPath')
    model.modelOutputPath = config.get('Main', 'modelOutputPath')
    
    model.startTraining(epochs)
    
    
    
    print "Done"