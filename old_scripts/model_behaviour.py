"""
Created by Flavio Martinelli at 11:03 05/03/2020
"""

import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras as keras

# My imports
from old_scripts.neuron_models_old import VLif, ILif, ILif3f
from plot_utils import raster_plot, v_plot

# set generic params
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # removes annoying warnings
tf.config.experimental_run_functions_eagerly(True) # allows debugging on @tf.function decorated funs

time_steps = 100

# One neuron test
spike_freq = 0.01
n_inputs = 3

inputs = np.zeros([1, time_steps, n_inputs])
inputs[0, (10, 30), 0] = 1.0
inputs[0, (40, 60), 1] = 1.0
inputs[0, (70, 90), 2] = 1.0

inputs = tf.Variable(inputs, dtype=tf.float32)

cell1 = VLif(n_rec=1, tau=10, thr=1, dt=1,
             w_in=[[5.0], [7.5], [7.5]],
             w_rec=[[0.0]],
             track_v=True)

cell2 = VLif(n_rec=1, tau=15, thr=1, dt=1,
             w_in=[[5.0], [7.5], [7.5]],
             w_rec=[[0.0]],
             track_v=True)

cell3 = VLif(n_rec=1, tau=20, thr=1, dt=1,
             w_in=[[5.0], [7.5], [7.5]],
             w_rec=[[0.0]],
             track_v=True)

rnn1 = keras.layers.RNN(cell1, return_sequences=True, return_state=False)
spk_out, v_out = tf.unstack(rnn1(inputs), num=2, axis=-1)

rnn2 = keras.layers.RNN(cell2, return_sequences=True, return_state=False)
spk_out2, v_out2 = tf.unstack(rnn2(inputs), num=2, axis=-1)

rnn3 = keras.layers.RNN(cell3, return_sequences=True, return_state=False)
spk_out3, v_out3 = tf.unstack(rnn3(inputs), num=2, axis=-1)

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
v_plot(ax1, v_out[0], out_spikes=spk_out[0], linewidth=1, color='b')
v_plot(ax1, v_out2[0], out_spikes=spk_out2[0], linewidth=1, color='g')
v_plot(ax1, v_out3[0], out_spikes=spk_out3[0], linewidth=1, color='m')
raster_plot(ax2, inputs[0], linewidth = 1)
ax1.set_title(f"Neurons with different taus")
ax1.legend([f'tau={cell1.tau.numpy()}, decay={cell1._decay}',
            f'tau={cell2.tau.numpy()}, decay={cell2._decay}',
            f'tau={cell3.tau.numpy()}, decay={cell3._decay}'])
ax1.locator_params(axis='y', nbins=25)
ax1.grid(which='minor')
ax1.grid()
plt.show()

# Different neuron type

cell1 = ILif(n_rec=1, tau_v=10, thr=1, dt=1,
             w_in=[[5.0], [7.5], [7.5]],
             w_rec=[[0.0]],
             track_v=True)

cell2 = ILif(n_rec=1, tau_v=15, thr=1, dt=1,
             w_in=[[5.0], [7.5], [7.5]],
             w_rec=[[0.0]],
             track_v=True)

cell3 = ILif(n_rec=1, tau_v=20, thr=1, dt=1,
             w_in=[[5.0], [7.5], [7.5]],
             w_rec=[[0.0]],
             track_v=True)

rnn1 = keras.layers.RNN(cell1, return_sequences=True, return_state=False)
spk_out, v_out = tf.unstack(rnn1(inputs), num=2, axis=-1)

rnn2 = keras.layers.RNN(cell2, return_sequences=True, return_state=False)
spk_out2, v_out2 = tf.unstack(rnn2(inputs), num=2, axis=-1)

rnn3 = keras.layers.RNN(cell3, return_sequences=True, return_state=False)
spk_out3, v_out3 = tf.unstack(rnn3(inputs), num=2, axis=-1)

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
v_plot(ax1, v_out[0], out_spikes=spk_out[0], linewidth=1, color='b')
v_plot(ax1, v_out2[0], out_spikes=spk_out2[0], linewidth=1, color='g')
v_plot(ax1, v_out3[0], out_spikes=spk_out3[0], linewidth=1, color='m')
raster_plot(ax2, inputs[0], linewidth = 1)
ax1.set_title(f"Neurons with different taus")
ax1.legend([f'tau={cell1.tau_v.numpy()}, decay={cell1._decay_v}',
            f'tau={cell2.tau_v.numpy()}, decay={cell2._decay_v}',
            f'tau={cell3.tau_v.numpy()}, decay={cell3._decay_v}'])
ax1.locator_params(axis='y', nbins=25)
ax1.grid()
plt.show()

# nice plot


# Different neuron type

cell1 = ILif(n_rec=1, tau_v=10, dt=1,
             w_in=[[5.0], [12.5], [10.5]],
             w_rec=[[0.0]], thr=0.8,
             track_v=True)

rnn1 = keras.layers.RNN(cell1, return_sequences=True, return_state=False)
spk_out, v_out = tf.unstack(rnn1(inputs), num=2, axis=-1)

f, (ax1) = plt.subplots(1, 1, figsize=(10, 5))
v_plot(ax1, v_out[0], out_spikes=None, linewidth=4, color='b')
ax1.hlines(0.8, 0, 100, colors=(1, 0, 0, 0.25), linestyles='--', linewidth=2)
#raster_plot(ax2, inputs[0], linewidth = 1)
ax1.locator_params(axis='y', nbins=25)
ax1.set_axis_off()
ax1.set_xlim([0, 57])
plt.show()


# one EPSP plot

inputs = np.zeros([1, time_steps, 1])
inputs[0, 10, 0] = 1.0

cell1 = ILif(n_rec=1, tau_v=10, thr=1, dt=1,
             w_in=[[10.0]],
             w_rec=[[0.0]],
             track_v=True)

rnn = keras.layers.RNN(cell1, return_sequences=True, return_state=False)

spk_out, v_out = tf.unstack(rnn(inputs), num=2, axis=-1)

f, (ax1) = plt.subplots(1, 1, figsize=(8, 5))
v_plot(ax1, v_out[0], out_spikes=None, linewidth=1, color='b')
ax1.locator_params(axis='y', nbins=25)
plt.show()



# one EPSP plot ILif3f

n_inputs = 2

inputs = np.zeros([1, time_steps, n_inputs])
inputs[0, (10, 30), 0] = 1.0
inputs[0, (40, 60), 1] = 1.0

cell1 = ILif3f(n_rec=2, tau_v=10, thr=1, dt=1,
             w_in=[[2.0, 1.0], [0.3, 3.0]],
             track_vars=True)

rnn = keras.layers.RNN(cell1, return_sequences=True, return_state=False)

out = rnn(inputs)

f, (ax1) = plt.subplots(1, 1, figsize=(8, 5))
v_plot(ax1, out['state'].v[0], out_spikes=out['state'].z[0], linewidth=1, color='b')
ax1.locator_params(axis='y', nbins=25)
ax1.grid()
plt.show()

