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


        nData = 500000

        #Create data and PDF
        Eta = getRandomEta(nData)
        Pt = getRandomPt(nData)
        #PDF = getEtaPDF(Eta)*getPtPDF(Pt)
        PDF = getPtPDF(Pt)
        labels = getLabels(PDF)
        #PDF = PDF / np.linalg.norm(PDF) #normalization
        #plt.scatter(Eta, PDF)
        #plt.scatter(getBinnedVar('Eta', 8), getFR_EtaHisto(Eta, labels, 8, nData))
        #plt.show()

        Eta = np.reshape(Eta, (-1,1))
        Pt = np.reshape(Pt, (-1,1))
        PDF = np.reshape(PDF, (-1,1))

        #features = np.hstack((PDF, Eta, Pt))
        features = np.hstack((PDF, Pt))

        np.random.shuffle(features)

        PDF = features[:,0]
        features = features[:,1:] #only pT and Eta are features
        labels = getLabels(PDF)


        print("Input data shape:",features.shape)

        self.numberOfFeatures = features.shape[1]

        assert features.shape[0] == labels.shape[0]

        self.features_placeholder = tf.placeholder(tf.float32)
        self.labels_placeholder = tf.placeholder(tf.float32)
        self.features = features
        self.labels = labels
        self.PDF = PDF
        self.Pt = Pt
        self.Eta = Eta
        self.nData = nData

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
