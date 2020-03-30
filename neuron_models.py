"""
Created by Flavio Martinelli at 11:08 26/02/2020
Inspired by https://github.com/IGITUGraz/LSNN-official
"""

import numpy as np
import numpy.random as rd
import tensorflow as tf
import tensorflow.keras as keras


class VLif(keras.layers.AbstractRNNCell):

    def __init__(self, n_rec, tau=20., thr=0.03, dt=1., dampening_factor=0.3,
                 dtype=tf.float32, w_in=None, w_rec=None, train_w_in=True, train_w_rec=True,
                 track_v=False):
        """
        Tensorflow cell object that simulates a LIF neuron with an approximation of the spike derivatives.
        :param n_rec: number of recurrent neurons
        :param tau: membrane time constant
        :param thr: threshold voltage
        :param dt: time step of the simulation
        :param dtype: data type of the cell tensors
        :param no_rec: cuts recurrent connections
        :param track_v: allows tracking of membrane voltage in the cell output
        """

        super(VLif, self).__init__()

        # Parameters init
        self.dt = dt
        self.n_rec = n_rec
        self.dampening_factor = dampening_factor
        self.data_type = dtype
        self._num_units = self.n_rec
        self.track_v = track_v
        self.w_in = w_in
        self.w_rec = w_rec
        self.train_w_in = train_w_in
        self.train_w_rec = train_w_rec

        self.tau = tf.Variable(tau, dtype=dtype, name="Tau", trainable=False)
        self._decay = tf.exp(-dt / tau)
        self.thr = tf.Variable(thr, dtype=dtype, name="Threshold", trainable=False)

    def build(self, input_shape):
        """ input_shape is structured as: [batch_size, time_steps, #input neurons] """
        n_in = input_shape[-1]

        if self.w_in is None:
            self.w_in = tf.Variable(rd.randn(n_in, self.n_rec) / np.sqrt(n_in), dtype=self.data_type,
                                    trainable=self.train_w_in, name="InputWeight")
        else:
            self.w_in = tf.Variable(self.w_in, dtype=self.data_type, trainable=self.train_w_in, name="InputWeight")

        if self.w_rec is None:
            self.w_rec = tf.Variable(rd.randn(self.n_rec, self.n_rec) / np.sqrt(self.n_rec), dtype=self.data_type,
                                     trainable=self.train_w_rec, name='RecurrentWeight')
        else:
            self.w_rec = tf.Variable(self.w_rec, dtype=self.data_type, trainable=self.train_w_rec, name='RecurrentWeight')

    @property
    def state_size(self):
        """ Shape of state corresponds to: [membrane voltage, spike state] """
        return self.n_rec, self.n_rec

    @property
    def output_size(self):
        return self.n_rec

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):

        if inputs is not None:
            batch_size = tf.shape(inputs)[0]

        v0 = tf.zeros(shape=(batch_size, self.n_rec), dtype=dtype)
        z0 = tf.zeros(shape=(batch_size, self.n_rec), dtype=dtype)

        return v0, z0

    @tf.function
    def call(self, inputs, state, scope=None, dtype=tf.float32):

        ''' Eager mode does not support named states therefore the LIFStateTuple arguments are:
        state.v = state[0]  membrane voltage
        state.z = state[1]  action potential
        '''

        ''' NOTE on tf.function: do not break a long line in multiple lines with '\', the character is
                                 not recognized by the AUTOGRAPH parser.'''

        # Compute inputs to the neurons
        synaptic_current = tf.matmul(inputs, self.w_in) + tf.matmul(state[1], self.w_rec)

        # Run LIF dynamics
        new_v, new_z, new_v_tmp = self.lif_dynamics(v=state[0], z=state[1], inp_spike_current=synaptic_current)
        new_state = new_v, new_z

        if self.track_v:  # returns also voltage trace tensor across the last dimension for compat with keras.model()
            # for better viz purposed when the neuron are firing a lot
            # return tf.stack([new_z, new_v_tmp], axis=-1), new_state
            return tf.stack([new_z, new_v], axis=-1), new_state
        else:
            return new_z, new_state

    def lif_dynamics(self, v, z, inp_spike_current):

        i_reset = z * self.thr * self.dt

        new_v_tmp = self._decay * v - i_reset
        new_v = new_v_tmp + (1 - self._decay) * inp_spike_current

        # Spike generation
        v_scaled = (new_v - self.thr) / self.thr

        new_z = SpikeFunction(v_scaled, self.dampening_factor)
        new_z = new_z * 1 / self.dt

        return new_v, new_z, new_v_tmp


class ILif(keras.layers.AbstractRNNCell):

    def __init__(self, n_rec, tau_v=20., tau_i=5., thr=0.03, dt=1., dampening_factor=0.3,
                 dtype=tf.float32, w_in=None, w_rec=None, train_w_in=True, train_w_rec=True,
                 track_v=False):
        """
        Tensorflow cell object that simulates a LIF neuron with an approximation of the spike derivatives.
        :param n_rec: number of recurrent neurons
        :param tau: membrane time constant
        :param thr: threshold voltage
        :param dt: time step of the simulation
        :param dtype: data type of the cell tensors
        :param no_rec: cuts recurrent connections
        :param track_v: allows tracking of membrane voltage in the cell output
        """

        super(ILif, self).__init__()

        # Parameters init
        self.dt = dt
        self.n_rec = n_rec
        self.dampening_factor = dampening_factor
        self.data_type = dtype
        self._num_units = self.n_rec
        self.track_v = track_v
        self.w_in = w_in
        self.w_rec = w_rec
        self.train_w_in = train_w_in
        self.train_w_rec = train_w_rec

        self.tau_v = tf.Variable(tau_v, dtype=dtype, name="Tau_v", trainable=False)
        self._decay_v = tf.exp(-dt / tau_v)
        self.tau_i = tf.Variable(tau_i, dtype=dtype, name="Tau_i", trainable=False)
        self._decay_i = tf.exp(-dt / tau_i)

        self.thr = tf.Variable(thr, dtype=dtype, name="Threshold", trainable=False)

    def build(self, input_shape):
        """ input_shape is structured as: [batch_size, time_steps, #input neurons] """

        n_in = input_shape[-1]

        if self.w_in is None:
            self.w_in = tf.Variable(rd.randn(n_in, self.n_rec) / np.sqrt(n_in), dtype=self.data_type,
                                    trainable=self.train_w_in, name="InputWeight")
        else:
            self.w_in = tf.Variable(self.w_in, dtype=self.data_type, trainable=self.train_w_in, name="InputWeight")

        if self.w_rec is None:
            self.w_rec = tf.Variable(rd.randn(self.n_rec, self.n_rec) / np.sqrt(self.n_rec), dtype=self.data_type,
                                     trainable=self.train_w_rec, name='RecurrentWeight')
        else:
            self.w_rec = tf.Variable(self.w_rec, dtype=self.data_type, trainable=self.train_w_rec, name='RecurrentWeight')

    @property
    def state_size(self):
        """ Shape of state corresponds to: [membrane voltage, synaptic current, spike state] """
        return self.n_rec, self.n_rec, self.n_rec

    @property
    def output_size(self):
        return self.n_rec

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):

        if inputs is not None:
            batch_size = tf.shape(inputs)[0]

        v0 = tf.zeros(shape=(batch_size, self.n_rec), dtype=dtype)
        i0 = tf.zeros(shape=(batch_size, self.n_rec), dtype=dtype)
        z0 = tf.zeros(shape=(batch_size, self.n_rec), dtype=dtype)

        return v0, i0, z0

    @tf.function
    def call(self, inputs, state, scope=None, dtype=tf.float32):

        ''' Eager mode does not support named states therefore the LIFStateTuple arguments are:
        state.v = state[0]  membrane voltage
        state.i = state[1]  synaptic current
        state.z = state[2]  action potential
        '''

        ''' NOTE on tf.function: do not break a long line in multiple lines with '\', the character is
                                 not recognized by the AUTOGRAPH parser.'''

        # Compute inputs to the neurons
        inp_spike_current = tf.matmul(inputs, self.w_in) + tf.matmul(state[1], self.w_rec)

        # Run LIF dynamics
        new_v, new_i, new_z = self.lif_dynamics(v=state[0], i=state[1], z=state[2],
                                                inp_spike_current=inp_spike_current)

        new_state = new_v, new_i, new_z

        if self.track_v:  # returns also voltage trace tensor across the last dimension for compat with keras.model()
            # for better viz purposed when the neuron are firing a lot
            # return tf.stack([new_z, new_v_tmp], axis=-1), new_state

            return tf.stack([new_z, new_v], axis=-1), new_state
        else:
            return new_z, new_state

    def lif_dynamics(self, v, i, z, inp_spike_current):

        with tf.name_scope('dynamics') as scope:

            i_reset = z * self.thr * self.dt

            new_i = self._decay_i * i + (1 - self._decay_i) * inp_spike_current

            new_v = self._decay_v * v + (1 - self._decay_v) * new_i - i_reset

            # Spike generation
            v_scaled = (new_v - self.thr) / self.thr

            new_z = SpikeFunction(v_scaled, self.dampening_factor)
            new_z = new_z * 1 / self.dt # ask guillaume about dt

        return new_v, new_i, new_z


class ILif_3flr(keras.layers.AbstractRNNCell):

    def __init__(self, n_rec, tau_v=20., tau_i=5., thr=0.6, dt=1., eta1= 0.001, eta2=0.015, theta = 0.31,
                 dtype=tf.float32, w_in=None, train_w_in=True, track_v_i=False):
        """
        Tensorflow cell object that simulates a LIF neuron with 3 factor learning rule. No backprop is used here.

        :param n_rec: number of recurrent neurons
        :param tau: membrane time constant
        :param thr: threshold voltage
        :param dt: time step of the simulation
        :param dtype: data type of the cell tensors
        :param track_v_i: allows tracking of membrane voltage and synaptic current in the cell output
        """

        super(ILif_3flr, self).__init__()

        # Parameters init
        self.dt = dt
        self.n_rec = n_rec
        self.data_type = dtype
        self._num_units = self.n_rec
        self.track_v_i = track_v_i
        self.w_in = w_in
        self.train_w_in = train_w_in
        self.eta1 = eta1
        self.eta2 = eta2
        self.theta = theta

        self.tau_v = tf.Variable(tau_v, dtype=dtype, name="Tau_v", trainable=False)
        self._decay_v = tf.exp(-dt / tau_v)
        self.tau_i = tf.Variable(tau_i, dtype=dtype, name="Tau_i", trainable=False)
        self._decay_i = tf.exp(-dt / tau_i)

        self.thr = tf.Variable(thr, dtype=dtype, name="Threshold", trainable=False)


    def build(self, input_shape):
        """ input_shape is structured as: [batch_size, time_steps, #input neurons] """

        n_in = input_shape[-1]

        if self.w_in is None:
            self.w_in = tf.Variable(rd.randn(n_in, self.n_rec) / np.sqrt(n_in), dtype=self.data_type,
                                    trainable=False, name="InputWeight")
        else:
            self.w_in = tf.Variable(self.w_in, dtype=self.data_type, trainable=self.train_w_in, name="InputWeight")

        self._clip_mask_max = np.ones_like(self.w_in.numpy()) * np.infty
        self._clip_mask_max[int(n_in/2):, :int(self.n_rec/2)] = 0.0
        self._clip_mask_max = tf.constant(self._clip_mask_max, dtype=self.data_type)

        self._clip_mask_min = -np.ones_like(self.w_in.numpy()) * np.infty
        self._clip_mask_min[int(n_in/2):, int(self.n_rec/2):] = 0.0
        self._clip_mask_min = tf.constant(self._clip_mask_min, dtype=self.data_type)

    @property
    def state_size(self):
        """ Shape of state corresponds to: [membrane voltage, synaptic current, spike state] """
        return self.n_rec, self.n_rec, self.n_rec

    @property
    def output_size(self):
        return self.n_rec

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):

        if inputs is not None:
            batch_size = tf.shape(inputs)[0]

        v0 = tf.zeros(shape=(batch_size, self.n_rec), dtype=dtype)
        i0 = tf.zeros(shape=(batch_size, self.n_rec), dtype=dtype)
        z0 = tf.zeros(shape=(batch_size, self.n_rec), dtype=dtype)

        return v0, i0, z0

    @tf.function
    def call(self, inputs, state, scope=None, dtype=tf.float32):

        ''' Eager mode does not support named states therefore the LIFStateTuple arguments are:
        state.v = state[0]  membrane voltage
        state.i = state[1]  synaptic current
        state.z = state[2]  action potential

        '''

        ''' NOTE on tf.function: do not break a long line in multiple lines with '\', the character is
                                 not recognized by the AUTOGRAPH parser.'''

        # Compute inputs to the neurons
        inp_spike_current = tf.matmul(inputs, self.w_in)

        # Run LIF dynamics
        new_v, new_i, new_z = self.lif_dynamics(v=state[0], i=state[1], z=state[2],
                                                inp_spike_current=inp_spike_current)

        # Compute update
        # TODO: to be tested
        # TODO: check with Martin the specific update details
        # pop_activity = (tf.reduce_sum(self.fi_filter(inp_spike_current[:, 0:int(self.n_rec/2)])) -
        #                 tf.reduce_sum(self.fi_filter(inp_spike_current[:, int(self.n_rec/2):])))
        pop_activity = tf.reduce_sum(self.fi_filter(new_v))
        tan = tf.math.tanh(pop_activity)
        surprise_factor = self.eta1 * tan + self.eta2 * tan * tf.cast(pop_activity > self.theta, dtype=self.data_type)

        # The input we need is only for the second population, zeroing inputs of first population to null their effect
        mask = tf.concat([tf.zeros([inputs.shape[0], int(inputs.shape[1]/2)]),
                          tf.ones([inputs.shape[0], int(inputs.shape[1]/2)])], axis=1)
        masked_input = tf.identity(inputs) * mask  # z^p_k in the equation of cosyne abstract

        # need to do outer product between inp_spike_curr and masked_input, consider using einsum bi,bk -> bik
        dw_ik = tf.einsum("bi,bk->ki", inp_spike_current, masked_input) * surprise_factor
        # TODO: consider that einsum performs summation over b (batches) which is not good
        # TODO: note that dw_ik is non zero only in a specific row for Maze.self._low_freq_p = 0.0

        self.w_in.assign_add(dw_ik)
        self.w_in.assign(tf.clip_by_value(self.w_in, self._clip_mask_min, self._clip_mask_max))

        # return interesting properties
        new_state = new_v, new_i, new_z

        # returns also voltage trace tensor across the last dimension for compat with keras.model(), if you want to
        # return a list of other params do not run the network in a keras model. Note that for running the cell in an
        # keras.layer.rnn, rnn() takes care of the batch dimension which is in the first axis (0th), everything that
        # needs to be returned as output of the cell must have the first axis as batch size (or dummy dimension)
        if self.track_v_i:
            # for better viz purposed when the neuron are firing a lot:
            # return tf.stack([new_z, new_v_tmp], axis=-1), new_state
            return [new_z, new_v, inp_spike_current,
                    tf.expand_dims(pop_activity, 0),
                    tf.expand_dims(surprise_factor, 0),
                    tf.expand_dims(dw_ik, 0)], new_state
        else:
            return new_z, new_state

    def lif_dynamics(self, v, i, z, inp_spike_current):

        with tf.name_scope('dynamics') as scope:

            i_reset = z * self.thr * self.dt

            new_i = self._decay_i * i + (1 - self._decay_i) * inp_spike_current

            new_v = self._decay_v * v + (1 - self._decay_v) * new_i - i_reset

            # Spike generation
            # new_z = tf.greater(new_v, self.thr)
            # new_z = tf.cast(new_z, dtype=tf.float32)

            new_z = self.stochastic_spike_function(new_v, tf.tanh)

            new_z = new_z * 1 / self.dt

        return new_v, new_i, new_z

    def stochastic_spike_function(self, v, fun=tf.tanh):
        """ Stochastic activation function of neuron based on a function of the voltage potential (e.g. fun = tf.tanh)
        """
        # Generate a random threshold for each neuron
        thresholds = rd.uniform(0., 1., size=v.shape)

        # Check if tanh(v) is above the generated threshold
        return tf.cast(tf.greater(fun(v), thresholds), dtype=self.data_type)

    @staticmethod
    def fi_filter(x, data_type=tf.float32):
        return tf.math.tanh(x) * tf.cast(x > 0., dtype=data_type)
        # return x * tf.cast(x > 0., dtype=data_type)

@tf.custom_gradient
def SpikeFunction(v_scaled, dampening_factor):
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, dtype=tf.float32)

    def grad(dy):
        dE_dz = dy
        dz_dv_scaled = tf.maximum(1 - tf.abs(v_scaled), 0)
        dz_dv_scaled *= dampening_factor

        dE_dv_scaled = dE_dz * dz_dv_scaled

        return [dE_dv_scaled,
                tf.zeros_like(dampening_factor)]

    return tf.identity(z_, name="SpikeFunction"), grad

