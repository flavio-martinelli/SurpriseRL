"""
Created by Flavio Martinelli at 11:08 26/02/2020
Inspired by https://github.com/IGITUGraz/LSNN-official
"""

import numpy as np
import numpy.random as rd
import tensorflow as tf
import tensorflow.keras as keras

from collections import OrderedDict, namedtuple


class Lif(keras.layers.AbstractRNNCell):

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

        super(Lif, self).__init__()

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


class Lif3f(keras.layers.Layer):

    def __init__(self, n_rec, tau_v=10., tau_i=5., dt=1., eta1= 0.00157, eta2=0.05749, theta=0.2206,
                 activity_filt_coeff=50, seed=666, dtype=tf.float32, w_in=None, track_vars=False):
        """
        Tensorflow cell object that simulates a LIF neuron with 3 factor learning rule. No backprop is used here.

        :param n_rec: number of recurrent neurons
        :param tau: membrane time constant
        :param thr: threshold voltage
        :param dt: time step of the simulation
        :param dtype: data type of the cell tensors
        :param track_vars: allows tracking of learning factors and internal variables, disable for faster computation
        """

        super(Lif3f, self).__init__()

        # Parameters init
        self.dt = dt
        self.n_rec = n_rec
        self.data_type = dtype
        self._num_units = self.n_rec
        self.track_vars = track_vars
        self.w_in = w_in
        self.eta1 = eta1
        self.eta2 = eta2
        self.theta = theta
        self.seed = seed
        self.activity_filt_coeff = activity_filt_coeff

        self.tau_v = tf.constant(tau_v, dtype=dtype, name="Tau_v")
        # self._decay_v = tf.exp(-dt / tau_v)
        self._decay_v = 0.9
        self.tau_i = tf.constant(tau_i, dtype=dtype, name="Tau_i")
        self._decay_i = tf.exp(-dt / tau_i)

        # Define state size for RNN to work
        self.state_size = (self.n_rec, self.n_rec, self.n_rec, 1)
        self.state_tuple = namedtuple('state_tuple', ('v', 'i', 'z', 'A'))


    def build(self, input_shape):
        """ input_shape is structured as: [batch_size, time_steps, #input neurons] """

        n_in = input_shape[-1]

        # Normalization constant for population activity
        self.A0 = 2 * self.n_rec / n_in

        if self.w_in is None:
            self.w_in = tf.Variable(rd.randn(n_in, self.n_rec) / np.sqrt(n_in), dtype=self.data_type, trainable=False,
                                    name="InputWeight")
        else:
            self.w_in = tf.Variable(self.w_in, dtype=self.data_type, name="InputWeight", trainable=False)

        self._clip_mask_max = np.ones_like(self.w_in.numpy()) * np.infty
        self._clip_mask_max[int(n_in/2):, :int(self.n_rec/2)] = 0.0
        self._clip_mask_max = tf.constant(self._clip_mask_max, dtype=self.data_type)

        self._clip_mask_min = -np.ones_like(self.w_in.numpy()) * np.infty
        self._clip_mask_min[int(n_in/2):, int(self.n_rec/2):] = 0.0
        self._clip_mask_min = tf.constant(self._clip_mask_min, dtype=self.data_type)

        self.rand_gen = tf.random.Generator.from_seed(self.seed)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):

        if inputs is not None:
            batch_size = tf.shape(inputs)[0]

        v0 = tf.zeros(shape=(batch_size, self.n_rec), dtype=dtype)
        i0 = tf.zeros(shape=(batch_size, self.n_rec), dtype=dtype)
        z0 = tf.zeros(shape=(batch_size, self.n_rec), dtype=dtype)
        A0 = tf.zeros(shape=(batch_size, 1), dtype=dtype)

        return self.state_tuple(v=v0, i=i0, z=z0, A=A0)

    @tf.function
    def call(self, inputs, state, scope=None, dtype=tf.float32):
        ''' Eager mode does not support named states therefore the LIFStateTuple arguments are indexed as in a list '''

        ''' NOTEç on tf.function: do not break a long line in multiple lines with '\', the character is
                                  not recognized by the AUTOGRAPH parser.'''

        # Compute inputs to the neurons
        inp_spike_current = tf.matmul(inputs, self.w_in)

        # Run LIF dynamics
        new_v, new_i, new_z = self.lif_dynamics(v=state[0], i=state[1], z=state[2],
                                                inp_spike_current=inp_spike_current)

        pop_activity = state[3] + (1/self.activity_filt_coeff) * (tf.reduce_sum(new_z) - state[3])
        tan = self.fi_filter(pop_activity / self.A0)
        surprise_factor = self.eta1 * tan + self.eta2 * tan * tf.cast(pop_activity / self.A0 > self.theta, dtype=self.data_type)

        # The input we need is only for the second population, zeroing inputs of first population to null their effect
        mask = tf.concat([tf.zeros([inputs.shape[0], int(inputs.shape[1]/2)]),
                          tf.ones([inputs.shape[0], int(inputs.shape[1]/2)])], axis=1)
        masked_input = tf.identity(inputs) * mask  # z^p_k in the equation of cosyne abstract

        # need to do outer product between inp_spike_curr and masked_input, consider using einsum bi,bk -> bik
        dw_ik = -tf.einsum("bi,bk->ki", inp_spike_current, masked_input) * surprise_factor
        # TODO: fix, einsum performs summation over b (batches)
        # note that dw_ik is non zero only in a specific row for Maze.self._low_freq_p = 0.0

        self.w_in.assign(tf.clip_by_value(self.w_in + dw_ik, self._clip_mask_min, self._clip_mask_max))

        # return interesting properties
        new_state = self.state_tuple(v=new_v, i=new_i, z=new_z, A=[pop_activity])

        out_dict = OrderedDict([('state', new_state),
                                ('syn_currents', inp_spike_current),
                                ('pop_activity', tf.expand_dims(pop_activity, 0)),
                                ('surprise', tf.expand_dims(surprise_factor, 0)),
                                ('dw_ik', tf.expand_dims(dw_ik, 0))])

        # returns also voltage trace tensor across the last dimension for compat with keras.model(), if you want to
        # return a list of other params do not run the network in a keras model. Note that for running the cell in an
        # keras.layer.rnn, rnn() takes care of the batch dimension which is in the first axis (0th), everything that
        # needs to be returned as output of the cell must have the first axis as batch size (or dummy dimension)
        if self.track_vars:
            return out_dict, new_state
        else:
            return new_z, new_state

    @tf.function
    def lif_dynamics(self, v, i, z, inp_spike_current):
        new_v = v + (1-self._decay_v) * (-v + inp_spike_current)
        new_z = self.stochastic_spike_function(new_v, tf.tanh)

        reset_mask = -new_z + 1
        if tf.reduce_sum(new_z) != 0.0:
            new_v *= reset_mask
        new_z = new_z * 1 / self.dt

        return new_v, new_v, new_z


    def stochastic_spike_function(self, v, fun=tf.tanh):
        """ Stochastic activation function of neuron based on a function of the voltage potential (e.g. fun = tf.tanh)
        """
        # Generate a random threshold for each neuron
        thresholds = tf.cast(self.rand_gen.uniform(v.shape, 0., 1.), dtype=self.data_type)

        # Check if tanh(v) is above the generated threshold
        return tf.cast(tf.greater(fun(v), thresholds), dtype=self.data_type)

    @staticmethod
    def fi_filter(x, data_type=tf.float32):
        return tf.math.tanh(x) * tf.cast(x > 0., dtype=data_type)


class Lif3fLoop:

    def __init__(self, n_obs, n_wm, n_err1, n_err2, tau_v=10., dt=1., eta1=0.00157, eta2=0.05749, theta=0.2206,
                 activity_filt_coeff=50, w_in=None, seed=666, dtype=tf.float32):
        """
        :param n_obs: Number of neurons in the observation input population
        :param n_wm: Number of neurons in the working memory input population
        :param n_err1: Number of neurons in the first error population (excitation from obs and inhibition from wm)
        :param n_err2: Number of neurons in the second error population (excitation from wm and inhibition from obs)
        :param tau_v: Neuron decay time constant (in dt units)
        :param dt: Timestep of simulation
        :param eta1: Slow learning rate (Ach)
        :param eta2: Fast learning rate (NE)
        :param theta: Threshold for activity that signals surprise
        :param activity_filt_coeff: Time filtering coefficient for error population activity
        :param w_in: Weight matrix of size [n_obs + n_wm, n_err1 + n_err2]
        :param seed: Random seed
        :param dtype: The data type used
        """

        # Parameters init
        self.n_obs = n_obs
        self.n_wm = n_wm
        self.n_err1 = n_err1
        self.n_err2 = n_err2
        self.tau_v = tau_v
        self._decay_v = tf.exp(-dt / tau_v)
        self.dt = dt
        self.eta1 = eta1
        self.eta2 = eta2
        self.theta = theta
        self.activity_filt_coeff = activity_filt_coeff
        self.w_in = w_in
        self.data_type = dtype

        # Random number generator
        self.rand_gen = tf.random.Generator.from_seed(seed)

        # Normalization constant for population activity
        self.A0 = 2 * (n_err1 + n_err2) / (n_obs + n_wm)

        assert self.w_in.shape == (n_obs + n_wm, n_err1 + n_err2), \
            f"Weight matrix shape should be {(n_obs + n_wm, n_err1 + n_err2)}, but instead is {self.w_in.shape}"

        self.w_in = tf.Variable(self.w_in, dtype=self.data_type, trainable=False)

        # Set clip matrices to avoid synapses switching signs
        self._clip_mask_max = np.ones_like(self.w_in.numpy()) * np.infty
        self._clip_mask_max[-n_wm:, :n_err1] = 0.0
        self._clip_mask_max = tf.constant(self._clip_mask_max, dtype=self.data_type)

        self._clip_mask_min = -np.ones_like(self.w_in.numpy()) * np.infty
        self._clip_mask_min[-n_wm:, -n_err2:] = 0.0
        self._clip_mask_min = tf.constant(self._clip_mask_min, dtype=self.data_type)



    @tf.function
    def run(self, inputs, v=None, activity=None):
        """ inputs: [time_steps, n_in]
        """
        t_steps = inputs.shape[0]
        if v is None:
            v = tf.zeros(self.n_err1 + self.n_err2)
        if activity is None:
            activity = tf.constant(0.0)

        # Generate thresholds (slightly faster but eats more memory))
        # thresholds = tf.cast(self.rand_gen.uniform(tf.TensorShape([t_steps]).concatenate(v.shape), 0., 1.),
        #                      dtype=self.data_type)

        v_trace = tf.TensorArray(self.data_type, size=t_steps)
        z_trace = tf.TensorArray(self.data_type, size=t_steps)
        # t_trace = tf.TensorArray(self.data_type, size=t_steps)
        activity_trace = tf.TensorArray(self.data_type, size=t_steps)
        surprise_trace = tf.TensorArray(self.data_type, size=t_steps)

        # The presynaptic input we need for plasticity is only coming from the working memory population,
        # masking inputs of observation population to null their effect

        mask = tf.concat([tf.zeros(self.n_obs), tf.ones(self.n_wm)], axis=0)

        """ NOTE: writing a python loop (range) inside tf.function (instead of using tf.range) forces tf.function to 
        build the unrolled graph of the loop. If that fits in memory the resulting simulation will be faster, whereas
        with rf.range the simulation is a little slower but way cheaper in terms of memory consumption. See 2nd answer:
        https://stackoverflow.com/questions/56547737/nested-tf-function-is-horribly-slow/61744937#61744937"""

        for t in range(t_steps):
            # Compute inputs to the neurons for all time_steps
            inp_spike_current = tf.linalg.matvec(self.w_in, inputs[t], transpose_a=True)

            # Run LIF dynamics
            v, z = self.lif_dynamics(v, inp_spike_current)

            activity = activity + (1/self.activity_filt_coeff) * (tf.reduce_sum(z) - activity)
            tan = self.fi_filter(activity / self.A0)
            surprise = self.eta1 * tan + self.eta2 * tan * tf.cast(activity / self.A0 > self.theta,dtype=self.data_type)

            masked_input = tf.identity(inputs[t]) * mask  # z^p_k in the equation of cosyne abstract

            # need to do outer product between inp_spike_curr and masked_input, consider using einsum bi,bk -> bki
            dw_ik = -tf.einsum("a,b->ba", inp_spike_current, masked_input) * surprise
            self.w_in.assign(tf.clip_by_value(self.w_in + dw_ik, self._clip_mask_min, self._clip_mask_max))

            v_trace = v_trace.write(t, v)
            z_trace = z_trace.write(t, z)
            # t_trace = t_trace.write(t, trh)
            activity_trace = activity_trace.write(t, activity)
            surprise_trace = surprise_trace.write(t, surprise)

        return v_trace.stack(), z_trace.stack(), activity_trace.stack(), surprise_trace.stack()

    @tf.function
    def run_no_plasticity(self, inputs, v=None):
        """ same as run() but no weight change is performed
        """
        t_steps = inputs.shape[0]
        if v is None:
            v = tf.zeros(self.n_err1 + self.n_err2)

        # Generate thresholds (slightly faster but eats more memory))
        # thresholds = tf.cast(self.rand_gen.uniform(tf.TensorShape([t_steps]).concatenate(v.shape), 0., 1.),
        #                      dtype=self.data_type)

        v_trace = tf.TensorArray(self.data_type, size=t_steps)
        z_trace = tf.TensorArray(self.data_type, size=t_steps)

        """ NOTE: writing a python loop (range) inside tf.function (instead of using tf.range) forces tf.function to 
        build the unrolled graph of the loop. If that fits in memory the resulting simulation will be faster, whereas
        with rf.range the simulation is a little slower but way cheaper in terms of memory consumption. See 2nd answer:
        https://stackoverflow.com/questions/56547737/nested-tf-function-is-horribly-slow/61744937#61744937"""

        for t in range(t_steps):
            # Compute inputs to the neurons for all time_steps
            inp_spike_current = tf.linalg.matvec(self.w_in, inputs[t], transpose_a=True)

            # Run LIF dynamics
            v, z = self.lif_dynamics(v, inp_spike_current)

            v_trace = v_trace.write(t, v)
            z_trace = z_trace.write(t, z)

        return v_trace.stack(), z_trace.stack()

    @tf.function
    def lif_dynamics(self, v, inp_spike_current):

        new_v = v + (1-self._decay_v) * (-v + inp_spike_current)
        new_z = self.stochastic_spike_function(new_v, tf.tanh)

        reset_mask = -new_z + 1.0
        new_v *= reset_mask

        return new_v, new_z

    def stochastic_spike_function(self, v, fun=tf.tanh):
        """ Stochastic activation function of neuron based on a function of the voltage potential (e.g. fun = tf.tanh)
        """
        # Generate a random threshold for each neuron
        thresholds = tf.cast(self.rand_gen.uniform(v.shape, 0., 1.), dtype=self.data_type)

        # Check if tanh(v) is above the generated threshold
        return tf.cast(tf.greater(fun(v), thresholds), dtype=self.data_type)

    @staticmethod
    def fi_filter(x, data_type=tf.float32):
        return tf.math.tanh(x) * tf.cast(x > 0., dtype=data_type)



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
