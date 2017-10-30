#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse

filename=""
image_size=128
num_channels=3
images=[]

#leggo immagine tramite opencv
image=cv2.imread(filename)
#resize the image
images.append(image)
images=np.array(images,dtype=np.uint8)
images=images.astype('float32')
images= np.multiply(images,1.0/255.0)
#l'input della rete richiede un file con 4 dimensioni in questo modo
x_batch=images.reshape(1,image_size,images,num_channels)

#Ripristino il modello salvato

sess= tf.Session()
#ricreo il grafico del modello
saver= tf.train.import_meta_graph("nome_del_graph.meta")
#carico i pesi e le variabili del modello
saver.restore(sess,tf.train.latest_checkpoint('./'))

#collegamento con il graph appena caricato
graph=tf.get_default_graph()

#riferimento all'operazione utilizzata per ottenere la classificazione in output
y_pred=graph.get_tensor_by_name("y_pred:0")

#fornisco l'immagine acquisita al placeholder di input della rete
x=graph.get_tensor_by_name("y_pred:0")
y_true=graph.get_tensor_by_name("y_true:0")
y_test_images= np.zeros((1,2))

#creo il feed_dict utilizzato per inserire i dati nella rete
feed_dict_testig={x:x_batch,y_true:y_test_images}

result= sess.run(y_pred,feed_dict=feed_dict_testig)
print (result)








