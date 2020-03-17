"""
Created by Flavio Martinelli at 11:27 26/02/2020
"""

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras
from neuron_models import VLif
from plot_utils import raster_plot, v_plot

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print('hello LCN hello')

n_epochs = 10
batch_size = 100
n_time = 250
n_neurons_in = 10
n_neurons_rec = 15

f0 = 0.1
dt = 1
spike_freq = dt * f0

metric = keras.metrics.MeanAbsoluteError()
optimizer = tf.keras.optimizers.Adam()

cell = VLif(n_rec=n_neurons_rec, track_v=True)
rnn = keras.layers.RNN(cell, return_sequences=True, return_state=False)

# with track_v activated the network output consists
# of a tuple (spikes [batch, t_step, n_rec], v_membrane [batch, t_step, n_rec])

model = keras.Sequential()
model.add(rnn)

inputs = tf.cast(tf.random.uniform([batch_size, n_time, n_neurons_in], 0, 1) < spike_freq, tf.float32)

spk_out, v_out = tf.unstack(model(inputs), num=2, axis=-1)

# PLOTTING BEFORE TRAINING

inputs = tf.cast(tf.random.uniform([batch_size, n_time, n_neurons_in], 0, 1) < spike_freq, tf.float32)
spk_out, v_out = tf.unstack(model(inputs), num=2, axis=-1)

ax = plt.subplot()
raster_plot(ax, inputs[0], linewidth = 1)
plt.show()

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))
v_plot(ax1, v_out[0], out_spikes=spk_out[0], linewidth=1, color='b')
raster_plot(ax2, inputs[0], linewidth = 1)
plt.show()

ax = plt.subplot()
raster_plot(ax, spk_out[0], linewidth=1)
plt.show()


def train_step(model, optimizer, inputs):
    with tf.GradientTape() as tape:
        spk_out, _ = tf.unstack(model(inputs), num=2, axis=-1)
        rate = tf.reduce_sum(spk_out, axis=(0, 1)) / (n_time * batch_size)
        loss = keras.losses.mean_squared_error(rate, f0)

    grads = tape.gradient(loss, rnn.trainable_variables)
    optimizer.apply_gradients(zip(grads, rnn.trainable_variables))

    return loss, metric(rate, f0)


def train(model, optimizer):
    step = 0
    loss = 0.0
    for e in range(n_epochs):
        inputs = tf.cast(tf.random.uniform([batch_size, n_time, n_neurons_in], 0, 1) < spike_freq, tf.float32)
        loss, mae = train_step(model, optimizer, inputs)
        tf.print('Step', step, ': loss', loss, '; MeanAbsoluteError', mae)
        step += 1
    return step, loss, mae

################################## GRAPH BUILDING #######################################

log_path = os.path.join(os.curdir,'tmp', 'mylogs')

writer = tf.summary.create_file_writer(log_path)

tf.summary.trace_on(graph=True, profiler=True)

train_step(model, optimizer, inputs)

with writer.as_default():
    tf.summary.trace_export(
      name="my_func_trace",
      step=0,
      profiler_outdir=os.path.join(log_path, 'profiler'))

########################################################################################

step, loss, mae = train(model, optimizer)

print('Final step', step, ': loss', loss, '; accuracy', mae)


#####################################################################################

# PLOTTING AFTER TRAINING

inputs = tf.cast(tf.random.uniform([batch_size, n_time, n_neurons_in], 0, 1) < spike_freq, tf.float32)
spk_out, v_out = tf.unstack(model(inputs), num=2, axis=-1)

ax = plt.subplot()
raster_plot(ax, inputs[0], linewidth=1)
plt.show()

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))
v_plot(ax1, v_out[0], out_spikes=spk_out[0], linewidth=1, color='b')
raster_plot(ax2, inputs[0], linewidth = 1)
plt.show()

ax = plt.subplot()
raster_plot(ax, spk_out[0], linewidth=1)
plt.show()

print('hellou')