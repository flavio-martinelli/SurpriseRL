"""
Created by Flavio Martinelli at 00:21 27/04/2020
"""

import os

import numpy as np
import tensorflow as tf
import numpy.random as rd
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime

from old_scripts.maze import Maze
from neuron_models import Lif3fLoop
from utils import average_w_matrix
from sklearn.preprocessing import normalize
from plot_utils import draw_matrix_tb

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

logdir = "logs/" + f"{n_in}+{n_pop}/" + "last/" + "d2/" + "thresh_norm/" + f"seed{seed}/" + datetime.now().strftime("%m%d-%H%M")
writer = tf.summary.create_file_writer(logdir)

# Initialize network [two input sets of size n_in and two output populations of size n_pop]
# Block identity matrix
obs_m1 = (1.0) * np.kron(np.eye(n_in), np.ones([1, out_in_ratio]))  #martin put them to 1 too
# wm_m1 = rd.uniform(1.5, 2., (n_in, n_pop))
wm_m1 = np.zeros((n_in, n_pop))
full_w_in = np.block([[obs_m1, -obs_m1], [-wm_m1, wm_m1]])

# net = SnnLoop(n_in=n_in*2, n_rec=n_pop*2, w_in=full_w_in, track_vars=True, eta1=eta1, eta2=eta2, theta=theta)
# net = Lif3fLoop(n_in=n_in*2, n_rec=n_pop*2, w_in=full_w_in, eta1=eta1, eta2=eta2, theta=theta)
net = Lif3fLoop(n_obs=n_in, n_wm=n_in, n_err1=n_pop, n_err2=n_pop, w_in=full_w_in, eta1=eta1, eta2=eta2, theta=theta)


# create observation and working memory input spike trains and concatenate them together [1, t_steps, n_in * 2]
spk_wm = tf.squeeze(Maze.spk_train_to_tensor(mz.generate_spike_train(mz.current_pos)))
new_pos, action = mz.apply_transition()
spk_obs = tf.squeeze(Maze.spk_train_to_tensor(mz.generate_spike_train(mz.current_pos)))
inputs = tf.concat([spk_obs, spk_wm], axis=-1)

a_trace = []
s_trace = []
f_trace = []
rnn_inner_state = None
# tf.profiler.experimental.server.start(6009)

last_v = None
last_activity = None


@tf.function
def run_step(inp, v, a):
    return net.run(inp, v, a)


for e in tqdm(range(epochs)):
    v, z, activity, surprise = run_step(inputs, last_v, last_activity)

    old_pos = new_pos
    new_pos, action = mz.apply_transition()

    spk_wm = spk_obs  # generate old position spike train again
    spk_obs = Maze.spk_train_to_tensor(mz.generate_spike_train(mz.current_pos))
    inputs = tf.concat([tf.squeeze(spk_obs), tf.squeeze(spk_wm)], axis=-1)

    last_v = v[-1]
    last_activity = activity[-1]

    # switch transition matrix
    if e % 1500 == 0 and e != 0:
        t_mat = mz.build_transition_matrix(dims=dims*2, symmetric=False, no_duplicates=True)
        # t_mat = mz.shuffle_maze()

    # Compute metrics
    a_trace.append(activity)
    s_trace.append(surprise)

    learned_t_mat = average_w_matrix(net.w_in*1.0, out_in_ratio)
    learned_t_mat = normalize(learned_t_mat[n_in:, n_in:], norm='l1', axis=0)
    f_trace.append(np.mean((t_mat-learned_t_mat)**2))

    with writer.as_default():
        tf.summary.scalar('frobenius_norm_transition_error', f_trace[-1], step=e)
        tf.summary.scalar('average_surprise', tf.reduce_mean(s_trace[-1]), step=e)
        tf.summary.scalar('average_activity', tf.reduce_mean(a_trace[-1]) / net.A0, step=e)

        if e % 500 == 0:
            tf.summary.image('learnt_transition_matrix', draw_matrix_tb(learned_t_mat), step=e)
            if e % 1500 == 0:
                tf.summary.image('true_transition_matrix', draw_matrix_tb(t_mat), step=e)


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
plt.pcolor(net.w_in.numpy())
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

_, ax = plt.subplots(1, 1)
plt.plot(np.array(a_trace))
plt.title('population activity')
plt.show()

_, ax = plt.subplots(1, 1)
plt.plot(np.array(f_trace))
plt.title('frobenius norm of transition matrix error')
plt.show()

print('theend')