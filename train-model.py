#!/usr/bin/env python
# -*- coding: utf-8 -*-
import dataset
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np

from numpy.random import seed
# seed(int i) i rappresenta il numero iniziale della sequenza che genera numeri casuali per inizializzare i valori
#della rete
seed(1)

from tensorflow import set_random_seed
set_random_seed(2)

#numero di elementi testati ogni volta
batch_size=32

classes=['dogs','cats']
num_classes=len(classes)

#parametri della rete:
#1) il 20% dei parametri verrà usato per la validazione

validation_size=0.2
img_size=128
num_channels=3
train_path='/Users/Eric/Desktop/eric/Programmazione/python/Tensorflow/training_data'

#carico tutte le immagini dal dataset

data= dataset.read_train_sets(train_path,img_size,classes,validation_size=validation_size)


print("Caricamento completato")
# richiama la proprietà della classe DataSet
print("Numero di file nel training set\t\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))

sess=tf.Session()
#i parametri di input e output dalla rete li definsco come placeholder

#definisco il tipo di inpu della rete
x= tf.placeholder(tf.float32,shape=[None,img_size,img_size,num_channels],name='x')

## labels
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

#parametri CNN

filter_size_conv1=3
num_filters_conv1= 32

filter_size_conv2=3
num_filters_conv2=32

filter_size_conv3=3
num_filters_conv3=64

fc_layer_size=128

#i weights e biases sono valori che cambiano il definsco come variabili
def create_weights(shape):
    #inzializzazione di una variabile utilizzando come una distribuzione normale con dev stand di 0.05
    #shape indica la dimensione del tensore di uscita
    #Una variabile al suo interno presenta un persistent tensor e i suoi valori sono accessibili attraverso session diverse
    return tf.Variable(tf.truncated_normal(shape,stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05,shape=[size]))

#Creazione della rete convoluzionale
def create_convolutional_layer(input,num_input_channels,conv_filter_size,num_filters):

    #definizione dei pesi
    weights=create_weights(shape=[conv_filter_size,conv_filter_size,num_input_channels,num_filters])
    #definzione dei bias
    biases=create_biases(num_filters)

    #convolutional layer
    layer=tf.nn.conv2d(input=input,
                       filter=weights,
                       strides=[1,1,1,1],
                       padding='SAME')

    layer+=biases

    layer= tf.nn.max_pool(value=layer,
                          ksize=[1,2,2,1],
                          strides=[1,2,2,1],
                          padding='SAME')
    #l'uscita del pooling è passata alla Relu per attivazione
    layer=tf.nn.relu(layer)

    return layer

#trasforma un vettore a 4 dimensioni in uno a 2 dimensioni necessario come input al fully connected layer
def create_flatten_layer(layer):

    layer_shape = layer.get_shape()
    #numero delle feature img_size*img_size*num_channels
    num_features = layer_shape[1:4].num_elements()

    ## Now, we Flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])

    return layer


def create_fc_layer(input,
                    num_inputs,
                    num_outputs,
                    use_relu=True):

    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # Fully connected layer input x e produce wx+b.
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

#CREAZIONE DELLA RETE NEURALE
layer_conv1 = create_convolutional_layer(input=x,
               num_input_channels=num_channels,
               conv_filter_size=filter_size_conv1,
               num_filters=num_filters_conv1)
layer_conv2 = create_convolutional_layer(input=layer_conv1,
               num_input_channels=num_filters_conv1,
               conv_filter_size=filter_size_conv2,
               num_filters=num_filters_conv2)

layer_conv3= create_convolutional_layer(input=layer_conv2,
               num_input_channels=num_filters_conv2,
               conv_filter_size=filter_size_conv3,
               num_filters=num_filters_conv3)


layer_flat=create_flatten_layer(layer_conv3)

#Fully connected layer
# get_shape() ritorna la dimensione statica del tensore
layer_fc1 = create_fc_layer(input=layer_flat,
                     num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                     num_outputs=fc_layer_size,
                     use_relu=True)

#la dimensione di output di questo layer è uguale al numero delle classi
layer_fc2 = create_fc_layer(input=layer_fc1,
                     num_inputs=fc_layer_size,
                     num_outputs=num_classes,
                     use_relu=False)

#uso la softmax per ottenere i valori delle classi predette con valori di probabilità
y_pred= tf.nn.softmax(layer_fc2,name='y_pred')
#con la funzione argmax scelgo solo la classe con il valore maggiore di probabilità
y_pred_cls = tf.argmax(y_pred, dimension=1)

#lancio la sessione di prova
sess.run(tf.global_variables_initializer())

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                    labels=y_true)
#calcolo il valore totale della funzione di costo
cost = tf.reduce_mean(cross_entropy)
#ottimizzatore per la backpropagation
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#inizializzazione delle variabili
sess.run(tf.global_variables_initializer())


#TRAINING E SALVATAGGIO DELLA RETE
def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    acc = sess.run(accuracy, feed_dict=feed_dict_train)
    val_acc = sess.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))

total_iterations = 0

#oggetto per salvare i valori della rete
saver= tf.train.Saver()

# training della rete
def train(num_iteration):
    global total_iterations

    for i in range(total_iterations,
                   total_iterations + num_iteration):

        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)

        feed_dict_tr = {x: x_batch,
                        y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch,
                         y_true: y_valid_batch}
        #training della rete con l'optimizer scelto inserendo i parametri tramite feed_dict
        sess.run(optimizer, feed_dict=feed_dict_tr)

        if i % int(data.train.num_examples / batch_size) == 0:
            val_loss = sess.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples / batch_size))

            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
            saver.save(sess, 'dogs-cats-model')

    total_iterations += num_iteration


train(num_iteration=3000)




