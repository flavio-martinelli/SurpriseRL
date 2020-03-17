"""
Created by Flavio Martinelli at 16:57 26/02/2020
"""

import tensorflow as tf
import numpy as np


def raster_plot(ax,spikes,linewidth=0.8,**kwargs):

    n_t, n_n = spikes.shape
    event_times, event_ids = np.where(spikes)
    max_spike = 10000
    event_times = event_times[:max_spike]
    event_ids = event_ids[:max_spike]

    for n, t in zip(event_ids,event_times):
        ax.vlines(t, n - 0.5, n + 0.5, linewidth=linewidth, **kwargs)

    ax.set_ylim([0 - .5, n_n - .5])
    ax.set_xlim([0, n_t])
    # ax.set_yticks([0, n_n])
    ax.locator_params(axis='y', nbins=n_n)
    ax.locator_params(axis='x', nbins=20)


def v_plot(ax, v_trace, out_spikes=None, linewidth=0.8, **kwargs):

    v_trace = v_trace.numpy().T
    max_v = np.max(v_trace)

    v_rest = np.array([idx*2*max_v for idx, v_neuron in enumerate(v_trace)])
    v_trace = np.array([v_neuron + idx*2*max_v for idx, v_neuron in enumerate(v_trace)])

    ax.hlines(v_rest, 0, v_trace.shape[1], colors=(0, 0, 0, 0.25), linestyles='--')
    ax.plot(v_trace.T, linewidth=linewidth, **kwargs)

    if out_spikes is not None:
        event_times, event_ids = np.where(out_spikes)
        for n, t in zip(event_ids, event_times):
            ax.vlines(t, n*2*max_v, n*2*max_v + max_v, linewidth=linewidth, colors='r')

    ax.set_xlim([0, v_trace.shape[1]])
    ax.locator_params(axis='x', nbins=20)
