from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.model_selection import KFold
from sklearn import preprocessing

import argparse
import os
import sys

import tensorflow as tf
import numpy as np
from dataManipulations import *
from plotUtilities import *
from model import *
from generateRandData import *

FLAGS = None

#deviceName = '/cpu:0'
#deviceName = '/:GPU:0'
deviceName = None
##############################################################################
##############################################################################
def makePlots(sess, myDataManipulations):
    #Fetch operations
    x = tf.get_default_graph().get_operation_by_name("input/x-input").outputs[0]
    y = tf.get_default_graph().get_operation_by_name("model/performance/Sigmoid").outputs[0]
    yTrue = tf.get_default_graph().get_operation_by_name("input/y-input").outputs[0]
    dropout_prob = tf.get_default_graph().get_operation_by_name("model/dropout_prob").outputs[0]
    trainingMode = tf.get_default_graph().get_operation_by_name("model/trainingMode").outputs[0]
    accuracy = tf.get_default_graph().get_operation_by_name("model/performance/accuracy/update_op").outputs[0]

    features = myDataManipulations.features
    featuresCopy = np.copy(features)
    labels = myDataManipulations.labels
    nData = myDataManipulations.nData
    #truePDF = myDataManipulations.PDF

    pT = features[:,1]
    Eta = features[:,0]

    result = sess.run([y, yTrue, accuracy], feed_dict={x: featuresCopy, yTrue: labels, dropout_prob: 0.0, trainingMode: False})
    modelResult = result[0]
    modelResult = np.reshape(modelResult,(1,-1))[0]

    modelResults = {"training": modelResult}

    print("Test sample accuracy:",result[2])

    indexesS = labels==1
    signalResponse = modelResult[indexesS]

    indexesB = labels==0
    backgroundResponse = modelResult[indexesB]

    plt.figure(1)
    plt.hist(signalResponse, bins = 20, label="fake tau")
    plt.legend(loc=2)
    plt.show(block=False)

    plt.figure(2)
    plt.hist(backgroundResponse, bins=20, label="true tau")
    plt.legend(loc=2)
    plt.show()

    print(labels.shape, pT.shape)

    nData = len(pT)
    nbins = 8
    pT_bins = getBinnedVar('pT', nbins)
    Eta_bins = getBinnedVar('Eta', nbins)
    trueFakesPt = getFR_PtHisto(pT, labels, nbins, nData)
    trueFakesEta = getFR_EtaHisto(Eta, labels, nbins, nData)

    print('pT bins: ', pT_bins.size, 'Eta bins: ', Eta_bins.size)

    #plot model prediction vs. input data histos
    plt.figure(1)
    plt.scatter(pT_bins, trueFakesPt, label = 'Input N_{fake}/N_{all}')
    plt.scatter(pT, modelResult, s=1, label = 'Model prediction')
    plt.xlabel('pT')
    plt.legend(loc=0)
    plt.show(block=False)

    plt.figure(2)
    plt.scatter(Eta_bins, trueFakesEta, label = 'Input N_{fake}/N_{all}')
    plt.scatter(Eta, modelResult, s=1, label = 'Model prediction')
    plt.xlabel('Eta')
    plt.legend(loc=0)
    plt.show()


##############################################################################
##############################################################################
def plot():

    with tf.Session(graph=tf.Graph()) as sess:

        print("Available devices:")
        devices = sess.list_devices()
        for d in devices:
            print(d.name)

        nEpochs = 1
        batchSize = 100
        nFolds = 2

        myDataManipulations = dataManipulations(nFolds, nEpochs, batchSize)

        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], FLAGS.model_dir)

        init_local = tf.local_variables_initializer()
        sess.run([init_local])

        makePlots(sess, myDataManipulations)
##############################################################################
##############################################################################
##############################################################################
def main(_):

  plot()
##############################################################################
##############################################################################
##############################################################################
if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--model_dir', type=str,
      default=os.path.join(os.getenv('PWD', './'),
                           'model'),
      help='Directory for storing model state')

  FLAGS, unparsed = parser.parse_known_args()

  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
##############################################################################
##############################################################################
##############################################################################
