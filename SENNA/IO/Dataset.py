class Dataset(object):
    
    def __init__(self, setX, setY, sentenceLengths, embeddingLookups, convMatrix, convFeatureExtractors):
        self.setX = setX
        self.setY = setY
        self.sentenceLengths = sentenceLengths
        self.featureExtractors = embeddingLookups
        
        self.convMatrix = convMatrix
        self.convFeatureExtractors = convFeatureExtractors