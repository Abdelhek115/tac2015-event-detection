# -*- coding: utf-8 -*-
import os
import numpy 
import json
import gzip

from EmbeddingsLookup import EmbeddingsLookup
from Vocabulary import Vocabulary
from Dataset import Dataset

import re
from unidecode import unidecode



class FeatureStore(object):
    replaceRules = [('u91', '['), ('u93', ']'), ('u45', '-'), ('u46', '.')]
    
    
    def __init__(self, config, fixedFeatureNames, convFeatureNames={}):
        self.config = config
        self.fixedFeatureNames = fixedFeatureNames
        self.convFeatureNames = convFeatureNames
        self.featureMapping = {} #Mapping feature name -> vocab path
        self.vocabPaths = set() #Paths of the different embeddings files
        self.vocabs = {} #Mapping vocab path -> Vocabulary 
        
        self.lookupCount = {} #Contains the number of performaned lookups per feature name
        self.lookupUnkownCount = {} #Contains the number of unknown embeddings
        
        self.featureMapping = {}
        
        self.labelTransform=None
        self.labelsToExcludeTrain=[]
        
        for feature in fixedFeatureNames:
            self.featureMapping[feature] = None
    
        for convFeatureName, convSubFeatures in convFeatureNames.iteritems():
            for subfeature in convSubFeatures:
                self.featureMapping[convFeatureName+"."+subfeature] = None
    
        self.minSentenceLength = 1
        
        numpy.random.seed(23) #Just random value for seeding
        
      
      
    def openFile(self, filename, mode):
        if filename.endswith('.gz'):
            return gzip.open(filename, mode)
        return open(filename, mode)
            
    
    def initVocabs(self): 
        """
        Initializes the vocabularies
        """       
        #1. Get the reduced feature names, without the [offest] information
        reducedFeatureNames = set()
        for featureName in self.featureMapping.keys():
            reducedFeatureNames.add(self.reduceFeatureName(featureName))
        

        #2. Get the path for the reduced feature names
        reducedFeatureNameLookup = {}
        self.vocabPaths = set();
        self.vocabsToInitialize = {}
        
        for featureName in reducedFeatureNames:
            vocabPath = self.getVocabPath(featureName)
            reducedFeatureNameLookup[featureName] = vocabPath
            self.vocabPaths.add(vocabPath)
                    
        #3. Lookup, featureName to vocab path
        for key in self.featureMapping.keys():
            self.featureMapping[key] = reducedFeatureNameLookup[self.reduceFeatureName(key)]
                        
        #4. Set the embeddings we like to update
        if self.config.has_option('Main', 'updateSingleFeatureEmbeddings'):
            self.updateEmbeddingsFeatureNames = set([feature.strip() for feature in self.config.get('Main', 'updateSingleFeatureEmbeddings').split(',')])
        else:
            self.updateEmbeddingsFeatureNames = set([])
        
    def getVocabPath(self, reducedFeatureName):  
        """
        Returns the path to the vocabulary. If 
        """
        vocabPath = os.path.join(self.config.get('Default', 'folder'), reducedFeatureName+'.vocab')
        vocabSize = self.config.get('Default', 'size')
        minimalCount = self.config.get('Default', 'minimalCount')
        initializeVocab = True
        
        if self.config.has_section(reducedFeatureName):
            if self.config.has_option(reducedFeatureName, 'path'):
                initializeVocab = False
                vocabPath = self.config.get(reducedFeatureName, 'path')
            
            if self.config.has_option(reducedFeatureName, 'size'):
                vocabSize = self.config.get(reducedFeatureName, 'size')   
                
            if self.config.has_option(reducedFeatureName, 'minimalCount'):
                minimalCount = self.config.get(reducedFeatureName, 'minimalCount')   
        
        vocabSize = int(vocabSize)
        minimalCount = int(minimalCount)          
        
        
        if initializeVocab:
            self.vocabsToInitialize[vocabPath] = (reducedFeatureName, vocabSize, minimalCount);
        
        return vocabPath
    
    
   
            
    def createDefaultVocabs(self, trainDataPath, fixedFeatureNames, convFeatureNames=None):
        reducedFeatures = []

        for vocabPath, item in self.vocabsToInitialize.iteritems():           
            reducedFeatures.append(item[0])
            
        # - Get the features we need to extract from the file -
        featuresToExtract = []
        for featureName in reducedFeatures:
            if featureName in fixedFeatureNames:
                featuresToExtract.append(featureName)
            else:
                featureAt0 = featureName+'[0]'
                if featureAt0 in fixedFeatureNames:
                    featuresToExtract.append(featureAt0) 
                else:
                    for fullFeatureName in featureName:
                        if self.reduceFeatureName(fullFeatureName) == featureName:
                            featuresToExtract.append(fullFeatureName) 
                            break
        
          
       
        if convFeatureNames != None:
            featuresToExtract += convFeatureNames.keys()
           
        
        
        
        # - Read file and extract the features -   
        featureSet = self.readFeatures(trainDataPath, featuresToExtract)
   
     
    
        # - Now initialize the vocab -
        for featureName in featuresToExtract:
            if featureName in fixedFeatureNames:
                vocabPath = self.featureMapping[featureName]
                _, vectorSize, minimalCount = self.vocabsToInitialize[vocabPath]
                
                print "%s, dimensions: %d, min-count: %d" % (featureName,vectorSize,minimalCount)
                self.createDefaultVocab(featureSet, featureName, vocabPath, vectorSize, minimalCount)
            elif featureName in convFeatureNames:
                convFeature = convFeatureNames[featureName]
           
                for subFeatureName in convFeature:
                    fullFeatureName = featureName+"."+subFeatureName
                    vocabPath = self.featureMapping[fullFeatureName]
                    
                    if vocabPath in self.vocabsToInitialize:
                        _, vectorSize, minimalCount = self.vocabsToInitialize[vocabPath]
                        print "%s, dimensions: %d, min-count: %d" % (fullFeatureName,vectorSize,minimalCount)
                    
                        self.createDefaultVocabForConvolution(featureSet, featureName, subFeatureName, vocabPath, vectorSize, minimalCount)
                
                    
        
        del featureSet
        
    def createDefaultVocab(self, featureSet, featureName, path, vectorSize, minimalCount):                
        """
        Initializes a new vocab file
        """        
        counts = {}
        for sentence in featureSet:
            for token in sentence: 
                value = token[1][featureName]
                
                if value not in counts:
                    counts[value] = 0
                        
                counts[value] += 1
                            
        self.writeDefaultVocab(counts, path, vectorSize, minimalCount)  
        
    def createDefaultVocabForConvolution(self, featureSet, featureName, subFeatureName, path, vectorSize, minimalCount):
        """
        Creates a default vocabulary from a convolution feature
        TODO: -- Make all vectors strictly positive as 0 is the min value of the max over time --
        """
        counts = {}
        for sentence in featureSet:
            for token in sentence: 
                subFeatureValues = json.loads(token[1][featureName])
               
                
                for subFeatures in subFeatureValues:
                    value = subFeatures[subFeatureName].encode("utf-8")         
                    
                    if value not in counts:
                        counts[value] = 0
                        
                    counts[value] += 1 
                
        self.writeDefaultVocab(counts, path, vectorSize, minimalCount, wordRange=0.1, paddingRange=0)  
    
    def writeDefaultVocab(self, counts, path, vectorSize, minimalCount, wordRange=0.1, paddingRange=0.1, unknownRange=0.1):      
        with open(path, 'w') as fOut:   
            containsPaddingVector = False
            containsUnkownVector = False
            
            #PADDING Vector   
            if not containsPaddingVector:
                vector = 2*paddingRange*numpy.random.rand(vectorSize)-paddingRange
                vectorStr = ' '.join(map(str, vector))
                fOut.write('%s %s\n' % ('PADDING', vectorStr)) 
        
            #UNKNOWN Vector   
            if not containsUnkownVector:
                vector = 2*unknownRange*numpy.random.rand(vectorSize)-unknownRange
                vectorStr = ' '.join(map(str, vector))
                fOut.write('%s %s\n' % ('UNKNOWN', vectorStr)) 
                
                 
            for key, value in counts.iteritems():
                if key == 'PADDING' or key == 'UNKNOWN':
                    continue
                
                if value >= minimalCount:                    
                    if ' ' in key:
                        print '%s contains illegal space character: %s' % (path, key)
                        continue
                    
                    vector = 2*wordRange*numpy.random.rand(vectorSize)-wordRange
                    vectorStr = ' '.join(map(str, vector))
                    fOut.write('%s %s\n' % (key, vectorStr))
                   
                    

        
                    
    def readFeatures(self, path, featuresToExtract = None, labelsToExclude=None):
        """
        Reads the features from CRFSuite format
        """
        features = []
        featureVecs = []   
        
        if featuresToExtract != None and len(featuresToExtract) == 0:
            return features     
        
        if labelsToExclude == None:
            labelsToExclude = self.labelsToExcludeTrain
        
        with self.openFile(path, 'r') as fIn:
            for line in fIn:
    
                line = line.strip()
                splits = line.split('\t')
               
                if len(line) == 0:
                    if len(featureVecs) > 0:
                        features.append(featureVecs)
                        featureVecs = []
                    continue
    
                label = intern(splits[0])
                
                if self.labelTransform != None:
                    label = self.labelTransform(label)
                
                if label in labelsToExclude:
                    continue
                
                featureVec = {}
                for i in xrange(1, len(splits)):
                    value = splits[i]
                    if value == '__BOS__' or value == '__EOS__':
                        continue
    
                    valueSplit = value.split('=',1)                
                
                    featureName = valueSplit[0]  
                    for rule in FeatureStore.replaceRules:
                        featureName = featureName.replace(rule[0], rule[1])             
                           
                    if featuresToExtract != None and featureName not in featuresToExtract: #Check if we want to extract this feature
                        continue
       
                    featureVec[intern(featureName)] = intern(valueSplit[1]) 
                
                featureVecs.append((label, featureVec))
        
        return features
        
           
           
    
    def reduceFeatureName(self, name):
        """
        Sanitizes the feature name
        """    
        if '[' in name:    
            name = name[0:name.find('[')]
        
        return name
    
    def loadVocabs(self):
        """
        Reads the suffixVocab files from disk an stores them in dictonaries and arries
        """
        self.vocabs = {}
        
        for path in self.vocabPaths:
            print path
            self.vocabs[path] = Vocabulary(path)
                
    def createMatrices(self, filePath, labelsMapping, labelsToExclude=[]):
        """
        This functions create the necessary matrices for training the neural network
        Y: Label (ids), 1 x n column matrix
        X: Features (word ids), m x n matrix
        lookupTables: Pointer (name) for the lookup table to map the word id to the vector
        sentenceLength: Sentence information
        @return: returns IO.Dataset
        """
        setX = []
        setConv = []
        setY = []
        sentenceLengths = []        
        fixedFeatures = self.fixedFeatureNames
        convFeatures = self.convFeatureNames
     
        
       
        self.lookupCount = {} #Contains the number of performaned lookups per feature name
        self.lookupUnkownCount = {} #Contains the number of unknown embeddings
        
        printWarning = False
        #1. Create X and Y matrices
        with self.openFile(filePath, 'r') as fIn:
            sentenceLength = 0
            
            for line in fIn:    
                line = line.strip()
                splits = line.split('\t')
               
                if len(line) == 0:  
                    if sentenceLength == 1:
                        if printWarning:
                            print "Warning: "+filePath+" sentence of length 1 found. In case of STC, Theano crashes for sentences of length 1. This single token is joined to the next sentence"
                            printWarning = False
                        continue
                    
                    if sentenceLength >= self.minSentenceLength:
                        sentenceLengths.append(sentenceLength)
                            
                    sentenceLength = 0                    
                    continue
    
                sentenceLength += 1
                
               
                label = intern(splits[0])
                
                if self.labelTransform != None:
                    label = self.labelTransform(label)
                
                if label in labelsToExclude:
                    continue
                
           
                
                fixedFeatureVector = self.mapFeatureValuesToIndices(splits[1:], fixedFeatures)
                if len(convFeatures) > 0:
                    convFeatureVector = self.mapConvFeatureValuesToIndices(splits[1:], convFeatures)
                    setConv.append(convFeatureVector)
                
                setX.append(fixedFeatureVector)
                setY.append(labelsMapping[label])
                
                               
 
            
        #3. Get the lookup tables for the fixedFeatures, store them in the fixed_feature_embeddings_lookup
        fixed_feature_embeddings_lookup = self.getFeatureExtractors(fixedFeatures) 
              
        #4. Map convolution layers to a fixed sized numpy matrix
        convMatrix = self.mapConvToMatrix(convFeatures, setConv)          
              
        #5. Get the lookup tables for the convFeatures
        convFeatureExtractors = self.getConvFeatureExtractors(convFeatures)
        
      
        
        return Dataset(setX=numpy.asarray(setX, dtype='int32'), setY=numpy.asarray(setY, dtype='int32'), 
                       sentenceLengths=sentenceLengths, embeddingLookups=fixed_feature_embeddings_lookup,
                       convMatrix = convMatrix,
                       convFeatureExtractors=convFeatureExtractors)
    
    def generateLabelsMapping(self, featureSet):
        """
        Generates a mapping of the labels to integers
        """
        setY = set()

        for sentence in featureSet:            
            for token in sentence:
                setY.add(token[0]) #Add the raw label
                
        labelsMapping = {}
        idx = 0
        for item in setY:
            labelsMapping[item] = idx
            idx += 1
            
            
        return labelsMapping
    
    def mapConvToMatrix(self, convFeatures, setConv):
        """
        Mapps the convolutional features to fixed sized values
        @param Dict<String,int[][]>[] setConv: A list of dictionaries with the convolutional layers features. Each dictionary mapps the name of the convolutional layer to a two dimensional array with the vocabulary indices  
        @param Dict<String, np.array>: Mapping convolutional feature name to fixed sized, 3 dimensional matrix with the feature values
        """
        convFeatureToMatrix = {}
        
        for convFeatureName in convFeatures:        
            maxNumFeatures = 0
            
            #Get the number of features
            for featureVectors in setConv:
                featuresForConv = featureVectors[convFeatureName]                
                maxNumFeatures = max(maxNumFeatures, len(featuresForConv))
                
            featureDimensions = len(convFeatures[convFeatureName])
            
            #Create a fixed sized fixedMatrix to store the values
            fixedMatrix = numpy.zeros((len(setConv), maxNumFeatures, featureDimensions), dtype='int32')
            
            #Copy the features to the fixed sized fixedMatrix
            for idx in xrange(len(setConv)):
                for idx2 in xrange(len(setConv[idx][convFeatureName])):
                    fixedMatrix[idx,idx2,:] = setConv[idx][convFeatureName][idx2]                    
                              
            convFeatureToMatrix[convFeatureName] = fixedMatrix                
                 
            
        return convFeatureToMatrix
        
        
                
                
    def mapFeatureValuesToIndices(self, features, featuresToExtract):
        """
        Maps a feature vector (a dictionariy of feature name + value) to the corresponding word indices
        Returns a vector with the word indices 
        @param String[] features: Array with Featurename=FeatureValue items
        @param String[] featuresToExtract: Array with feature names
        """
        featureVector = {}
        for featureEntry in features:
            if featureEntry == '__BOS__' or featureEntry == '__EOS__':
                continue
            
            featureSplit = featureEntry.split('=',1)                
            
            featureName = featureSplit[0]
            for rule in FeatureStore.replaceRules:
                featureName = featureName.replace(rule[0], rule[1])                
   
            if featureName in featuresToExtract:
                featureVector[featureName] = featureSplit[1] 
        
        
        wordIndices = []
        
        for featureName in featuresToExtract:           
            wordIndices.append(self.getWordIndex(featureName, featureVector[featureName]))
        
        return wordIndices
    
    def mapConvFeatureValuesToIndices(self, features, featuresToExtract):
        """
        Mapps convolutional features to their according word indices
        @param String[] features: Array with Featurename=FeatureValue items
        @param Dict<String,String[]> featureVector: dict with ConvFeatureName -> String[] {Subfeature names}}
        @return Dict<String,Int[][]>: Returns a dict that maps to the vocab indices for all sub features
        """
        featureVector = {}
        for featureEntry in features:
            if featureEntry == '__BOS__' or featureEntry == '__EOS__':
                continue
            
            featureSplit = featureEntry.split('=',1)                
            
            featureName = featureSplit[0]
            for rule in FeatureStore.replaceRules:
                featureName = featureName.replace(rule[0], rule[1])                
   
            if featureName in featuresToExtract:
                featureVector[featureName] = featureSplit[1] 
        
        
        wordIndices = {}
        
        for featureName, subFeatures in featuresToExtract.iteritems():
            wordIndices[featureName] = []  
            subFeatureVectors =  json.loads(featureVector[featureName])  
            
            for subFeatureVector in subFeatureVectors:
                wordIndicesSubFeature = []
                for subFeatureName in subFeatures:
                    fullFeatureName = featureName+"."+subFeatureName                      
                    wordIndicesSubFeature.append(self.getWordIndex(fullFeatureName, subFeatureVector[subFeatureName].encode("utf-8")))
                
                wordIndices[featureName].append(wordIndicesSubFeature)
      
        return wordIndices
        
    
    
    def getWordIndex(self, featureName, featureValue):
        """
        Returns the index for the given feature value
        """
        vocabPath = self.featureMapping[featureName]
        wordIndices = self.vocabs[vocabPath].getWordIndices()
        
        if featureName not in self.lookupCount:
            self.lookupCount[featureName] = 0
        self.lookupCount[featureName] += 1
        
        if featureValue in wordIndices:
            return wordIndices[featureValue]
        elif featureValue.lower() in wordIndices: #Check if we find a lower cased version
            #print "Lower: "+featureValue
            return wordIndices[featureValue.lower()]
        elif self.normalizeWord(featureValue) in wordIndices:
            normWord = self.normalizeWord(featureValue)
            #print "normWord: %s -> %s" % (featureValue, normWord)
            return wordIndices[normWord]
        else:           
            if featureName not in self.lookupUnkownCount:
                self.lookupUnkownCount[featureName] = 0
            self.lookupUnkownCount[featureName] += 1
        
            #if featureName == 'affix4.text':
            #    print "Unknown: "+featureValue
            return wordIndices['UNKNOWN']
        
        
    def multiple_replacer(self, key_values):
        #replace_dict = dict(key_values)
        replace_dict = key_values
        replacement_function = lambda match: replace_dict[match.group(0)]
        pattern = re.compile("|".join([re.escape(k) for k, v in key_values.iteritems()]), re.M)
        return lambda string: pattern.sub(replacement_function, string)
    
    def multiple_replace(self, string, key_values):
        return self.multiple_replacer(key_values)(string)

    def normalizeWord(self, line):         
        line = unicode(line, "utf-8") #Convert to UTF8
        line = line.replace(u"„", u"\"")
      
        line = line.lower(); #To lower case
        
        #Replace all special charaters with the ASCII corresponding, but keep Umlaute
        #Requires that the text is in lowercase before
        replacements = dict(((u"ß", "SZ"), (u"ä", "AE"), (u"ü", "UE"), (u"ö", "OE")))
        replacementsInv = dict(zip(replacements.values(),replacements.keys()))
        line = self.multiple_replace(line, replacements)
        line = unidecode(line)
        line = self.multiple_replace(line, replacementsInv)
        
        line = line.lower() #Unidecode might have replace some characters, like € to upper case EUR
        
        line = re.sub("([0-9][0-9.,]*)", '0', line) #Replace digits by NUMBER        
   
      
        return line.strip();
        
    def getFeatureExtractors(self, features):
        """
        Returns a list with the corresponding embeddings lookup feature extractors required by Theano
        """
        featureExtractors = []
        for featureName in features:
            vocabPath = self.featureMapping[featureName]
            vocab = self.vocabs[vocabPath]
            embeddingSize = vocab.getEmbeddingSize()
            sharedEmbeddingsTable = vocab.getSharedEmbeddings()
            update = (featureName in self.updateEmbeddingsFeatureNames)
            featureExtractors.append( EmbeddingsLookup(featureName, embeddingSize, sharedEmbeddingsTable, update) )
        
        return featureExtractors
    
    def getConvFeatureExtractors(self, features):
        """
        Returns a list with the corresponding embeddings lookup feature extractors required by Theano for convolutional features
        """
        convFeatureExtractors = {}
        for featureName, subFeatures in features.iteritems():
            subFeatureExtractors = []
            for subFeature in subFeatures:
                fullFeatureName = featureName+"."+subFeature
                
                vocabPath = self.featureMapping[fullFeatureName]
                vocab = self.vocabs[vocabPath]
                embeddingSize = vocab.getEmbeddingSize()
                sharedEmbeddingsTable = vocab.getSharedEmbeddings()
                update = (fullFeatureName in self.updateEmbeddingsFeatureNames)             
                subFeatureExtractors.append( EmbeddingsLookupForConv(fullFeatureName, embeddingSize, sharedEmbeddingsTable, update) )
            
            convFeatureExtractors[featureName] = subFeatureExtractors
        
        return convFeatureExtractors
    
    def saveUpdatedEmbeddings(self, epoch=0):
        if self.config.has_option('Main', 'updateSingleFeatureEmbeddings') and self.config.has_option('Main', 'folderUpdatedEmbeddings'):            
            updateEmbeddings = set([feature.strip() for feature in self.config.get('Main', 'updateSingleFeatureEmbeddings').split(',')])
            updateEmbeddings = updateEmbeddings.intersection(self.fixedFeatureNames)
           
            for featureName in updateEmbeddings:
                vocabPath = self.featureMapping[featureName]
                vocab = self.vocabs[vocabPath]                
                
                self.storeUpdatedVocab(vocab, epoch)
        
        if self.config.has_option('Main', 'updateVocabulariesForFeatures') and self.config.has_option('Main', 'folderUpdatedEmbeddings'):
            for vocab in self.vocabs.itervalues():
                if vocab.updateSharedEmbeddings:     
                    self.storeUpdatedVocab(vocab, epoch)               
                    
                    
    def storeUpdatedVocab(self, vocab, epoch):
        sharedEmbeddingsTable = vocab.getSharedEmbeddings()
        embeddings = sharedEmbeddingsTable.get_value();
        vocabPath = vocab.vocabPath
        
        wordIndices = self.vocabs[vocabPath].getWordIndices()
        vocabName = vocab.getVocabName();
        
        newPath = self.config.get('Main', 'folderUpdatedEmbeddings')+'/'+vocabName+'-'+str(epoch)+'.vocab'
        
        fOut = open(newPath, 'w')
        for word, wordIdx in wordIndices.iteritems():
            embedding = embeddings[wordIdx]
            fOut.write('%s %s\n' % (word, ' '.join(map(str, embedding))))
            
        fOut.close()
            
    def getVocabForFeature(self, featureName):
        reducedName = self.reduceFeatureName(featureName)
        vocabPath = self.getVocabPath(reducedName)
        return self.vocabs[vocabPath]
        
        
        
        