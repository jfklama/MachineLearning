import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from generateRandData import *

##############################################################################
##############################################################################
##############################################################################
class dataManipulations:

    def getNumpyMatricesFromRawData(self):

        #legs, jets, global_params, properties = pd.read_pickle(self.fileName)

        #print("no of legs: ", len(legs))
        #print("no of jets: ", len(jets))
        #print("global params: ", global_params.keys())
        #print("object properties:",properties.keys())

        nData = 50000

        #Create data and PDF
        Eta = getRandomEta(nData)
        Pt = getRandomPt(nData)
        PDF = getEtaPDF(Eta)+getPtPDF(Pt)
        PDF = PDF / np.linalg.norm(PDF)

        #plt.plot( Pt, PDF )
        #plt.show()

        #print(PDF)

        #np.random.shuffle(Eta)
        #np.random.shuffle(Pt)
        #np.random.shuffle(PDF)


        Eta = np.reshape(Eta, (-1,1))
        Pt = np.reshape(Pt, (-1,1))
        PDF = np.reshape(PDF, (-1,1))

        '''
        Eta = np.transpose(Eta)
        Pt = np.transpose(Pt)
        PDF = np.transpose(PDF)
        '''

        self.Eta = Eta
        self.Pt = Pt

        features = np.hstack((PDF, Eta, Pt))

        #Select events with MET>10
        #index = met[:,0]>10
        #features = features[index]

        '''
        index = features[:,0]<250
        features = features[index]

        index = features[:,0]>50
        features = features[index]
        '''
        '''
        index = features[:,0]>85
        features = features[index]

        index = features[:,0]<95
        features = features[index]
        '''

        np.random.shuffle(features)

        PDF = features[:,0]
        features = features[:,1:]

        labels = PDF
        labels[ labels >= 0.004 ] = 1
        labels[ labels < 0.004 ] = 0
        #labels = np.reshape(labels, (-1,1))
        #np.transpose(labels)
        #labels = np.hstack((labels))

        #print(Pt[:10], Eta[:10], PDF[:10], labels[:10])
        print(labels)

        print("Input data shape:",features.shape)

        self.numberOfFeatures = features.shape[1]

        assert features.shape[0] == labels.shape[0]

        self.features_placeholder = tf.placeholder(tf.float32)
        self.labels_placeholder = tf.placeholder(tf.float32)
        #self.fastMTT = fastMTT
        self.features = features
        self.labels = labels
        self.PDF = PDF

    def makeCVFoldGenerator(self):

        foldSplitter = KFold(n_splits=self.nFolds)
        self.foldsIndexGenerator = foldSplitter.split(self.labels, self.features)
        self.indexList = list(enumerate(self.foldsIndexGenerator))

    def makeDatasets(self):

        aDataset = tf.data.Dataset.from_tensor_slices((self.features_placeholder, self.labels_placeholder))
        self.trainDataset = aDataset.batch(self.batchSize)
        self.trainDataset = self.trainDataset.repeat(self.nEpochs)

        aDataset = tf.data.Dataset.from_tensor_slices((self.features_placeholder, self.labels_placeholder))
        self.validationDataset = aDataset.batch(10000)



    def getDataIteratorAndInitializerOp(self, aDataset):

        aIterator = tf.data.Iterator.from_structure(aDataset.output_types, aDataset.output_shapes)
        init_op = aIterator.make_initializer(aDataset)
        return aIterator, init_op

    def getCVFold(self, sess, aFold):

        if(aFold>=len(self.indexList)):
            print("Fold too big: ",aFold," number of folds is ",self.nFolds)
            return None

        trainIndexes = self.indexList[aFold][1][1]
        validationIndexes = self.indexList[aFold][1][0]

        self.numberOfBatches = np.ceil(len(trainIndexes)/self.batchSize)
        self.numberOfBatches = (int)(self.numberOfBatches)

        foldFeatures = self.features[trainIndexes]
        foldLabels = self.labels[trainIndexes]
        feed_dict={self.features_placeholder: foldFeatures, self.labels_placeholder: foldLabels}
        sess.run(self.trainIt_InitOp, feed_dict=feed_dict)

        foldFeatures = self.features[validationIndexes]
        foldLabels = self.labels[validationIndexes]
        feed_dict={self.features_placeholder: foldFeatures, self.labels_placeholder: foldLabels}
        sess.run(self.validationIt_InitOp, feed_dict=feed_dict)

        return self.trainIterator.get_next(), self.validationIterator.get_next()

    def __init__(self, nFolds, nEpochs, batchSize):
        #self.fileName = fileName
        self.batchSize = batchSize
        self.nFolds = nFolds
        self.nEpochs = nEpochs
        #self.smearMET = smearMET

        self.getNumpyMatricesFromRawData()
        self.makeCVFoldGenerator()
        self.makeDatasets()

        self.trainIterator, self.trainIt_InitOp = self.getDataIteratorAndInitializerOp(self.trainDataset)
        self.validationIterator, self.validationIt_InitOp = self.getDataIteratorAndInitializerOp(self.validationDataset)

##############################################################################
##############################################################################
##############################################################################
def makeFeedDict(sess, dataIter):
    aBatch = sess.run(dataIter)
    x = aBatch[0]
    y = np.reshape(aBatch[1],(-1,1))
    return x, y
##############################################################################
##############################################################################
##############################################################################
