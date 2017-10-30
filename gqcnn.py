#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import tensorflow as tf
import urllib2


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys

from datasets import imagenet
from nets import inception_v1
from preprocessing import inception_preprocessing

slim = tf.contrib.slim

image_size=inception_v1.inception_v1.default_image_size
print image_size


with tf.Graph().as_default():
    #descrivo le operazioni effettuate dentro al graph
    #url= sys.argv[1] passo il comando da terminale
    #url="https://static.pexels.com/photos/356378/pexels-photo-356378.jpeg"
    url="http://www.ilsecoloxix.it/rf/Image-lowres_Multimedia/IlSecoloXIXWEB/genova/foto/2016/10/04/funghiporcini-H161004191833.jpg"
    image_string=urllib2.urlopen(url).read()
    #trasforma l'immagine in un tensor adatto per essere processato in tf
    image= tf.image.decode_jpeg(image_string,channels=3)
    processed_image=inception_preprocessing.preprocess_image(image,image_size,image_size,is_training=False)
    processed_images=tf.expand_dims(processed_image,0)
    print tf.shape(processed_images)

    with slim.arg_scope(inception_v1.inception_v1_arg_scope()):
        logits,_= inception_v1.inception_v1(processed_images,num_classes=1001,is_training=False)
    probabilities= tf.nn.softmax(logits)
    checkpoints_dir='slim_pretrained'
    #prende i valori delle variabili dal modello precedentemete trainato
    #print os.path.join(checkpoints_dir,'inception_v1.ckpt')

    init_fn=slim.assign_from_checkpoint_fn(
        "/Users/Eric/Desktop/cv-tricks/Tensorflow-tutorials/Tensorflow-slim-run-prediction/slim_pretrained/inception_v1.ckpt",
        slim.get_variables_to_restore())

    with tf.Session() as sess:
        init_fn(sess)
        np_image,probabilities= sess.run([image,probabilities])
        probabilities=probabilities[0,0:]
        sorted_inds=[i[0] for i in sorted(enumerate(-probabilities),key=lambda x:x[1])]

    names= imagenet.create_readable_names_for_imagenet_labels()
    result_text=''

    for i in range(5):
        index=sorted_inds[i]
        print('Probability %0.2f%% => [%s]' % (100 * probabilities[index], names[index]))
    result_text += str(names[sorted_inds[0]]) + '=>' + str(
        "{0:.2f}".format(100 * probabilities[sorted_inds[0]])) + '%\n'
    plt.figure(num=1, figsize=(8, 6), dpi=80)
    plt.imshow(np_image.astype(np.uint8))
    plt.text(150, 100, result_text, horizontalalignment='center', verticalalignment='center', fontsize=21, color='blue')
    # plt.text(200,600,result_text)
    plt.axis('off')
    plt.show()












