"""
Created by Flavio Martinelli at 16:57 26/02/2020
"""

import tfmpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


def raster_plot(ax,spikes,linewidth=0.8, ybins=16, **kwargs):

    n_t, n_n = spikes.shape
    event_times, event_ids = np.where(spikes)

    for n, t in zip(event_ids, event_times):
        ax.vlines(t, n - 0.5, n + 0.5, linewidth=linewidth, **kwargs)

    ax.set_ylim([0 - .5, n_n - .5])
    ax.set_xlim([0, n_t])
    # ax.set_yticks([0, n_n])
    ax.locator_params(axis='y', nbins=ybins)
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


@tfmpl.figure_tensor
def draw_matrix_tb(matrix, figsize=(4,4)):
    """ Matrix plotting for tensorboard """
    fig = tfmpl.create_figure(figsize=figsize)
    ax = fig.add_subplot(111)
    im = ax.pcolor(matrix)
    ax.invert_yaxis()
    fig.colorbar(im)
    fig.tight_layout()

    return fig


def annotate_yrange(ymin, ymax,
                    label=None,
                    offset=-0.1,
                    width=-0.1,
                    ax=None,
                    patch_kwargs={'facecolor':'white'},
                    line_kwargs={'color':'black'},
                    text_kwargs={'rotation':-90}
                    ):

    if ax is None:
        ax = plt.gca()

    # x-coordinates in axis coordinates,
    # y coordinates in data coordinates
    trans = transforms.blended_transform_factory(
        ax.transAxes, ax.transData)

    # a bar indicting the range of values
    rect = Rectangle((offset, ymin), width=width, height=ymax-ymin,
                     transform=trans, clip_on=False, **patch_kwargs)
    ax.add_patch(rect)

    # delimiters at the start and end of the range mimicking ticks
    min_delimiter = Line2D((offset+width, offset), (ymin, ymin),
                           transform=trans, clip_on=False, **line_kwargs)
    max_delimiter = Line2D((offset+width, offset), (ymax, ymax),
                           transform=trans, clip_on=False, **line_kwargs)
    ax.add_artist(min_delimiter)
    ax.add_artist(max_delimiter)

    # label
    if label:
        x = offset + 0.5 * width
        y = ymin + 0.5 * (ymax - ymin)
        # we need to fix the alignment as otherwise our choice of x
        # and y leads to unexpected results;
        # e.g. 'right' does not align with the minimum_delimiter
        ax.text(x, y, label,
                horizontalalignment='center', verticalalignment='center',
                clip_on=False, transform=trans, **text_kwargs)
