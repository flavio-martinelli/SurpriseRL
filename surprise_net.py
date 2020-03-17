"""
Created by Flavio Martinelli at 10:23 09/03/2020
"""

import numpy as np
import tensorflow as tf
import numpy.random as rd
import tensorflow.keras as keras
import matplotlib.pyplot as plt

from maze import Maze
from neuron_models import VLif, ILif
from plot_utils import raster_plot, v_plot

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

n_in = 16
n_pop = 128
t_steps = 100
dt = 1

init_weights = 0.3 * rd.randn(n_in, n_pop) / np.sqrt(n_in)
init_weights_r = np.zeros((n_pop , n_pop))

cell = ILif(n_rec=n_pop, w_in=init_weights, w_rec=init_weights_r, track_v=True, train_w_rec=False, train_w_in=False)
rnn = keras.layers.RNN(cell, return_sequences=True)

mz = Maze()
t_mat = mz.build_transition_matrix(dims=2)
mz.set_spike_train_params(t_steps=t_steps, high_freq_p=.9, low_freq_p=0.1)

print(f"Starting position: {mz.current_pos}")

_, ax = plt.subplots(1, 1)
plt.pcolor(t_mat)
plt.gca().invert_yaxis()
plt.colorbar()
plt.show()

for i in range(5):

    # Generate spike train for current position
    spk_train = mz.generate_spike_train(mz.current_pos)

    # Build tensor, expand dims on axis 0 to have a batch size of 1 and match rnn shape requirements
    spk_train_tensor = tf.expand_dims(tf.constant(mz.flatten_spk_train(spk_train), dtype=tf.float32), axis=0)

    spk_out, v_out = tf.unstack(rnn(spk_train_tensor), num=2, axis=-1)

    print(f"New position: {mz.apply_transition()}")

    # f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))
    # v_plot(ax1, v_out[0], out_spikes=spk_out[0].numpy(), linewidth=1, color='b')
    # raster_plot(ax2, spk_train.reshape(t_steps, -1), linewidth = 1)
    # ax1.xaxis.grid(linestyle='-.')  # vertical lines
    # plt.show()

