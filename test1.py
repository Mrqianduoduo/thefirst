# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 09:10:49 2018

@author:qianduoduo
"""
import tensorflow as tf
import numpy as np

xdata=np.random.rand(100).astype(np.float32)
ydata=xdata*0.1+0.3

#creat tensorflow structure start 
Weights=tf.Variable(tf.random_uniform([1],-1.0,1.0))
bias=tf.Variable(tf.zeros([1]))


y=Weights*xdata+bias

loss=tf.reduce_mean(tf.square(y-ydata))
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss)

init=tf.initialize_all_variables()
#creat tensorflow structure start 

sess=tf.Session()
sess.run(init)    #critical

for step in range(601):
    sess.run(train)
    if step%20==0:
        print(step,sess.run(Weights),sess.run(bias))