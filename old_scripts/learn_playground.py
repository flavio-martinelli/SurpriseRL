"""
Created by Flavio Martinelli at 15:17 12/03/2020
"""
import os

import numpy as np
import tensorflow as tf
import numpy.random as rd
import tensorflow.keras as keras
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime

from old_scripts.maze import Maze
from old_scripts.neuron_models_old import ILif3f
from utils import average_w_matrix
from sklearn.preprocessing import normalize
from plot_utils import raster_plot, v_plot, draw_matrix_tb

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# tf.config.experimental_run_functions_eagerly(True)

theta = 0.3169981156555285
eta1 = 0.0009796194970225012
eta2 = 0.015360662582262801

dims = 2

seed = 635
tf.random.set_seed(seed)
rd.seed(seed)

t_steps = 100
epochs = 1500*6
out_in_ratio = 2**3

# Initialize maze
mz = Maze((4, 4))
t_mat = mz.build_transition_matrix(dims=dims, symmetric=True)
mz.set_spike_train_params(t_steps, high_freq_p=1.0, low_freq_p=0.0)

n_in = mz.tot_room_number
n_pop = mz.tot_room_number*out_in_ratio
dt = 1

logdir = "logs/" + f"{n_in}+{n_pop}/" + "old/" + "d2/" + "thresh_norm/" + f"seed{seed}/" + datetime.now().strftime("%m%d-%H%M")
writer = tf.summary.create_file_writer(logdir)

# Initialize network [two input sets of size n_in and two output populations of size n_pop]
# Block identity matrix
obs_m1 = (1.0) * np.kron(np.eye(n_in), np.ones([1, out_in_ratio]))  #martin put them to 1 too
# wm_m1 = rd.uniform(0.0, 1., (n_in, n_pop))
wm_m1 = np.zeros((n_in, n_pop))
full_w_in = np.block([[obs_m1, -obs_m1], [-wm_m1, wm_m1]])

cell = ILif3f(n_rec=n_pop * 2, w_in=full_w_in, track_vars=True, eta1=eta1, eta2=eta2, theta=theta)
rnn = keras.layers.RNN(cell, return_sequences=True)

# create observation and working memory input spike trains and concatenate them together [1, t_steps, n_in * 2]
spk_wm = Maze.spk_train_to_tensor(mz.generate_spike_train(mz.current_pos))
mz.apply_transition()
spk_obs = Maze.spk_train_to_tensor(mz.generate_spike_train(mz.current_pos))
inputs = tf.concat([spk_obs, spk_wm], axis=-1)

a_trace = []
s_trace = []
f_trace = []

rnn_inner_state = cell.get_initial_state(inputs=None, batch_size=1, dtype=cell.dtype)

# tf.profiler.experimental.server.start(6009)


@tf.function
def run_step(inp, inner_state):
    return rnn(inp, initial_state=inner_state)


for e in tqdm(range(epochs)):
    # spk_out, v_out, inp_spike_current, pop_activity, surprise_factor, dw_ik = rnn(inputs)

    out = run_step(inputs, rnn_inner_state)

    # out = rnn(inputs, initial_state=rnn_inner_state)
    # plt.plot(np.squeeze(out['state'].A))
    # plt.show()

    mz.apply_transition()
    spk_wm = spk_obs  # generate old position spike train again
    spk_obs = Maze.spk_train_to_tensor(mz.generate_spike_train(mz.current_pos))
    inputs = tf.concat([spk_obs, spk_wm], axis=-1)

    rnn_inner_state = cell.state_tuple(v=out['state'].v[:,-1,:],
                                       i=out['state'].i[:,-1,:],
                                       z=out['state'].z[:,-1,:],
                                       A=out['state'].A[0][:,-1])

    # switch transition matrix
    if e % 1500 == 0 and e != 0:
        t_mat = mz.build_transition_matrix(dims=dims*2, symmetric=False, no_duplicates=True)

    # Compute metrics
    a_trace.append(tf.squeeze(out['state'].A))
    s_trace.append(tf.squeeze(out['surprise']))

    learned_t_mat = average_w_matrix(cell.w_in / mz.tot_room_number, out_in_ratio)
    learned_t_mat = normalize(learned_t_mat[n_in:, n_in:], norm='l1', axis=1)
    f_trace.append(np.mean((t_mat-learned_t_mat)**2))
    # f_trace.append(tf.math.square(tf.norm(t_mat - learned_t_mat, ord='fro', axis=(0, 1))).numpy())

    with writer.as_default():
        tf.summary.scalar('frobenius_norm_transition_error', f_trace[-1], step=e)
        tf.summary.scalar('average_surprise', tf.reduce_mean(s_trace[-1]), step=e)
        tf.summary.scalar('average_activity', tf.reduce_mean(a_trace[-1]) / cell.A0, step=e)

        if e % 500 == 0:
            tf.summary.image('learnt_transition_matrix', draw_matrix_tb(learned_t_mat), step=e)
            if e % 1500 == 0:
                tf.summary.image('true_transition_matrix', draw_matrix_tb(t_mat), step=e)


f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 25))
v_plot(ax1, out['state'].v[0], out_spikes=out['state'].z[0].numpy(), linewidth=1, color='b')
raster_plot(ax2, inputs[0], linewidth=1)
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
plt.pcolor(full_w_in)
plt.gca().invert_yaxis()
plt.colorbar()
plt.show()

_, ax = plt.subplots(1, 1)
plt.pcolor(cell.w_in.numpy())
plt.gca().invert_yaxis()
plt.colorbar()
plt.show()

_, ax = plt.subplots(1, 1)
plt.pcolor(learned_t_mat)
plt.gca().invert_yaxis()
plt.colorbar()
plt.show()

_, ax = plt.subplots(1, 1)
plt.plot(np.array(s_trace).mean(axis=1))
plt.title('average surprise per epoch')
plt.show()
#
# _, ax = plt.subplots(1, 1)
# plt.plot(np.array(a_trace))
# plt.title('population activity')
# plt.show()

_, ax = plt.subplots(1, 1)
plt.plot(np.array(f_trace))
plt.title('frobenius norm of transition matrix error')
plt.xscale('log')
plt.show()

print('theend')