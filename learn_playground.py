import numpy as np
import tensorflow as tf
import numpy.random as rd
import tensorflow.keras as keras
import matplotlib.pyplot as plt

from tqdm import tqdm
from maze import Maze
from neuron_models import ILif_3flr
from plot_utils import raster_plot, v_plot



import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

t_steps = 100
epochs = 10000

# Initialize maze
mz = Maze((4, 4))
t_mat = mz.build_transition_matrix(dims=2, symmetric=True)
mz.set_spike_train_params(t_steps)

n_in = mz.tot_room_number
n_pop = mz.tot_room_number*2
dt = 1

# Initialize network [two input sets of size n_in and two output populations of size n_pop]
obs_m1 = mz.tot_room_number*(1./mz.tot_room_number) * np.kron(np.eye(n_in), np.ones([1, int(n_pop/n_in)]))  # Block identity matrix
wm_m1 = 10 * rd.randn(n_in, n_pop) / np.sqrt(n_pop)
full_w_in = np.block([[obs_m1, -obs_m1], [wm_m1, -wm_m1]])

cell = ILif_3flr(n_rec=n_pop*2, w_in=full_w_in, track_v_i=True)
rnn = keras.layers.RNN(cell, return_sequences=True)

# create observation and working memory input spike trains and concatenate them together [1, t_steps, n_in * 2]
spk_wm = Maze.spk_train_to_tensor(mz.generate_spike_train(mz.current_pos))
mz.apply_transition()
spk_obs = Maze.spk_train_to_tensor(mz.generate_spike_train(mz.current_pos))
inputs = tf.concat([spk_obs, spk_wm], axis=-1)

for e in tqdm(range(epochs), ):
    spk_out, v_out, inp_spike_current, pop_activity, surprise_factor, dw_ik = rnn(inputs)
    mz.apply_transition()
    spk_wm = spk_obs
    spk_obs = Maze.spk_train_to_tensor(mz.generate_spike_train(mz.current_pos))
    inputs = tf.concat([spk_obs, spk_wm], axis=-1)

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))
v_plot(ax1, v_out[0], out_spikes=spk_out[0].numpy(), linewidth=1, color='b')
raster_plot(ax2, inputs[0], linewidth = 1)
ax1.xaxis.grid(linestyle='-.')  # vertical lines
plt.show()

# dw = tf.reduce_sum(tf.squeeze(dw_ik), axis=0)
# plt.pcolor(dw)
# plt.show()

_, ax = plt.subplots(1, 1)
plt.pcolor(t_mat)
plt.gca().invert_yaxis()
plt.colorbar()
plt.show()

_, ax = plt.subplots(1, 1)
plt.pcolor(cell.w_in.numpy())
plt.gca().invert_yaxis()
plt.colorbar()
plt.show()

print('theend')