from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
import numpy as np
from dataManipulations import *
from plotUtilities import *
from model import *

FLAGS = None

#deviceName = '/cpu:0'
#deviceName = '/:GPU:0'
deviceName = None

##############################################################################
##############################################################################
##############################################################################
def runCVFold(sess, iFold, myDataManipulations, myTrainWriter, myValidationWriter):
    #Fetch operations
    x = tf.get_default_graph().get_operation_by_name("input/x-input").outputs[0]
    y = tf.get_default_graph().get_operation_by_name("model/performance/Sigmoid").outputs[0]

    yTrue = tf.get_default_graph().get_operation_by_name("input/y-input").outputs[0]

    dropout_prob = tf.get_default_graph().get_operation_by_name("model/dropout_prob").outputs[0]
    trainingMode = tf.get_default_graph().get_operation_by_name("model/trainingMode").outputs[0]

    train_step = tf.get_default_graph().get_operation_by_name("model/train/Adam")

    loss = tf.get_default_graph().get_operation_by_name("model/train/total_loss").outputs[0]
    lossL2 = tf.get_default_graph().get_operation_by_name("model/train/get_regularization_penalty").outputs[0]
    accuracy = tf.get_default_graph().get_operation_by_name("model/performance/accuracy/update_op").outputs[0]

    mergedSummary = tf.get_default_graph().get_operation_by_name("monitor/Merge/MergeSummary").outputs[0]

    aTrainIterator, aValidationIterator = myDataManipulations.getCVFold(sess, iFold)
    numberOfBatches = myDataManipulations.numberOfBatches
    accuracyValue = 0

    #Train
    iBatch = -1
    iEpoch = 0
    while True:
        try:
            xs, ys = makeFeedDict(sess, aTrainIterator)
            iBatch+=1
            iEpoch = (int)(iBatch/numberOfBatches)

            sess.run([train_step], feed_dict={x: xs, yTrue: ys, dropout_prob: FLAGS.dropout, trainingMode: True})

            #Evaluate training performance
            if(iEpoch%10==0 and iBatch%numberOfBatches==0):
                result = sess.run([mergedSummary, accuracy, lossL2, loss], feed_dict={x: xs, yTrue: ys, dropout_prob: 0.0, trainingMode: False})
                iStep = iEpoch + iFold*FLAGS.max_epoch
                trainSummary = result[0]
                modelLoss = result[3]
                myTrainWriter.add_summary(trainSummary, iStep)
                print("Epoch number:",iEpoch,
                      "total loss:",modelLoss,
                      "Train accuracy:", result[1],
                      "regularisation loss",result[2])

        except tf.errors.OutOfRangeError:
            break
    #########################################
    #Evaluate performance on validation data
    try:
        xs, ys = makeFeedDict(sess, aValidationIterator)
        result = sess.run([accuracy, mergedSummary],
                        feed_dict={x: xs, yTrue: ys, dropout_prob: 0.0, trainingMode: False})
        accuracyValue = result[0]
        validationSummary = result[1]
        iStep = (iFold+1)*FLAGS.max_epoch - 1
        myValidationWriter.add_summary(validationSummary, iStep)

        print("Validation. Fold:",iFold,
              "Epoch:",iEpoch,
              "Accuracy:",accuracyValue)

        result = sess.run([y, yTrue], feed_dict={x: xs, yTrue: ys, dropout_prob: 0.0, trainingMode: False})
        modelResult = result[0]
        labels = result[1]
        plotDiscriminant(modelResult, labels, "Validation")
    except tf.errors.OutOfRangeError:
        print("OutOfRangeError")

    return accuracyValue
##############################################################################
##############################################################################
##############################################################################
def train():

    sess = tf.Session()

    print("Available devices:")
    devices = sess.list_devices()
    for d in devices:
        print(d.name)

    #nFolds = 2         #data split into equal training and validation parts
    nFolds = 5          #5 folds with 4 hidden layers works faster
    nEpochs = FLAGS.max_epoch
    #batchSize = 128
    batchSize = 4096
    myDataManipulations = dataManipulations(nFolds, nEpochs, batchSize)
    numberOfFeatures = myDataManipulations.numberOfFeatures
    #nNeurons = [numberOfFeatures, 128, 128]
    nNeurons = [numberOfFeatures, 32, 32, 32, 32]

    # Input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, name='x-input')
        yTrue = tf.placeholder(tf.float32, name='y-input')

    with tf.name_scope('model'):
        myModel = Model(x, yTrue, nNeurons, FLAGS.learning_rate, FLAGS.lambda_lagrange)

    #initialize global and local (for accuracy measurement) variables
    init_global = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()
    sess.run([init_global, init_local])

    # Merge all the summaries and write them out to
    with tf.name_scope('monitor'):
        merged = tf.summary.merge_all()
    myTrainWriter = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    myValidationWriter = tf.summary.FileWriter(FLAGS.log_dir + '/validation', sess.graph)
    ###############################################
    '''
    ops = tf.get_default_graph().get_operations()
    for op in ops:
        print(op.name)
    '''
    ###############################################
    '''
    sess.run(init)
    iFold = 0
    runCVFold(sess, iFold, myDataManipulations, myTrainWriter, myValidationWriter)

    ###########################################
    '''
    accuracyTable = np.array([])
    lossTable = np.array([])

    for iFold in range(0, 1):
        sess.run([init_global, init_local])
        aAccuracy = runCVFold(sess, iFold, myDataManipulations, myTrainWriter, myValidationWriter)
        accuracyTable = np.append(accuracyTable, aAccuracy)

    print("Mean accuracy: %0.2f 95CL: (%0.2f - %0.2f)" % (accuracyTable.mean(),
                                                             accuracyTable.mean()-2*accuracyTable.std(),
                                                             accuracyTable.mean()+2*accuracyTable.std()))
    ###########################################


    myTrainWriter.close()
    myValidationWriter.close()
    # Save the model to disk.
    y = tf.get_default_graph().get_operation_by_name("model/output/Identity").outputs[0]

    tf.saved_model.simple_save(sess, FLAGS.model_dir,
                               inputs={"x": x, "yTrue": yTrue},
                               outputs={"y": y})
    print("Model saved in file: %s" % FLAGS.model_dir)
    return
##############################################################################
##############################################################################
##############################################################################
def main(_):

  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)

  if tf.gfile.Exists(FLAGS.model_dir):
    tf.gfile.DeleteRecursively(FLAGS.model_dir)

  train()
##############################################################################
##############################################################################
if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--max_epoch', type=int, default=50,
                      help='Number of epochs')

  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')

  parser.add_argument('--lambda_lagrange', type=float, default=0.1,
                      help='Largange multipler for L2 loss')

  parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')

  parser.add_argument('--model_dir', type=str,
      default=os.path.join(os.getenv('PWD', './'),
                           'model'),
      help='Directory for storing model state')

  parser.add_argument('--log_dir', type=str,
      default=os.path.join(os.getenv('PWD', './'),
                           'logs'),
      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()

  '''
    parser.add_argument('--train_data_file', type=str,
        default=os.path.join(os.getenv('PWD', './'),
                             'data/htt_features_train.pkl'),
        help='Directory for storing training data')
  '''

  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
##############################################################################
##############################################################################
