#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

sess= tf.InteractiveSession()

#crea un tensore utilizzabile in tf
y_hat= tf.convert_to_tensor(np.array([[0.5,1.5,0.1],[2.2,1.3,1.7]]))

#trasofrmo il vettore con la funzione softmax che indica le probabilità per ogni riga
# array([[ 0.227863  ,  0.61939586,  0.15274114],
#        [ 0.49674623,  0.20196195,  0.30129182]])

#TABELLA DELLE PROBABILITA
#                      Pr(Class 1)  Pr(Class 2)  Pr(Class 3)
 #                   ,--------------------------------------
#Training instance 1 | 0.227863   | 0.61939586 | 0.15274114
#Training instance 2 | 0.49674623 | 0.20196195 | 0.30129182

#   1 METODO

y_hat_softmax= tf.nn.softmax(y_hat)

#vettore che contiene le classificazioni corrette dei miei dati di input
y_true = tf.convert_to_tensor(np.array([[0.0, 1.0, 0.0],[0.0, 0.0, 1.0]]))

#calcolo i loss per ogni Training instance FUNZIONE DI CROSS ENTROPY=y_true*tf.log(y_hat_softmax)
loss_instance_1= -tf.reduce_sum(y_true*tf.log(y_hat_softmax),reduction_indices=[1])
sess.run(loss_instance_1)
#calcolo il valore della funzione di costo totale non per il singolo training
total_loss=tf.reduce_mean(-tf.reduce_sum(y_true* tf.log(y_hat_softmax),reduction_indices=[1]))
sess.run(total_loss)

#   SECONDO METODO

#logits normalizza i dati senza bisogno di preprocessarli con la funzione softmax


#uso la funzione tf.nn.softmax_cross_entropy_with_logits per calcolare direttmente il valore della fz costo
total_loss_digits= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_hat,y_true))
sess.run(total_loss_digits)



#STAMPA DEL TENSORE
sess.run(y_hat)
#permette di stampare il valore di un tensore a terminale
y_hat=tf.Print(y_hat,[y_hat])

b=tf.add(y_hat,y_hat).eval()