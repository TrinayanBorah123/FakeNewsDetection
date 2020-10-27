# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 22:03:32 2020

@author: Trinayan Borah
"""

import numpy as np
import tensorflow.keras.models
from tensorflow.keras.models import model_from_json
import tensorflow as tf

def init(): 
	json_file = open('model.json','r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	#load woeights into new model
	loaded_model.load_weights("model.h5")
	print("Loaded Model from disk")

	#compile and evaluate loaded model
	loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	#loss,accuracy = model.evaluate(X_test,y_test)
	#print('loss:', loss)
	#print('accuracy:', accuracy)
	#graph = tf.get_default_graph()
    
	return loaded_model
    #return loaded_model