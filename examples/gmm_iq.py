"""
gmm_iqp.py - predict state from iq data

Author: tpr0p

Refs:
[0] https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
#sklearn.mixture.GaussianMixture.fit
[1] https://towardsdatascience.com/gaussian-mixture-models-explained-6986aaf5a95
[2] https://github.com/davidavdav/GaussianMixtures.jl
"""

import json
import os
import random

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


# paths
# WDIR = working directory, this is the repo root
WDIR = os.environ.get("SLAB_PATH", "../")
# DDIR = data directory, for you this might be "S:\\"
DDIR = "/media/slab"
# Path to directory of file of interest. For windows, replace "/" with "\\".
DATA_PATH = "_Data/200814 - 3DMM2 cooldown 11 - sideband with LO and mixer - StimEm/data/"
SAVE_PATH = os.path.join(WDIR, "out")
PLOT_FILE_PATH = os.path.join(SAVE_PATH, "iqp.png")

# plotting
DPI = 300
DPI_FINAL = int(1e3)

# constants
COMPONENT_COUNT = 3
VALIDATION_COUNT = 4
MAX_ITER = 100
GLABEL = 0
ELABEL = 1
FLABEL = 2

"""
grab_data - read some iq data, tagged by state, from a file into memory
"""
def grab_data():
    expt_name = "histogram"
    filelist = [0]
    for jj, i in enumerate(filelist):
        filename = os.path.join(
            DDIR, DATA_PATH, "{:05d}_{}.h5".format(i, expt_name.lower())
        )
        with h5py.File(filename,"r") as a:
            hardware_cfg =  (json.loads(a.attrs['hardware_cfg']))
            experiment_cfg =  (json.loads(a.attrs['experiment_cfg']))
            quantum_device_cfg =  (json.loads(a.attrs['quantum_device_cfg']))
            ran = hardware_cfg['awg_info']['keysight_pxi']['m3102_vpp_range']
            expt_cfg = (json.loads(a.attrs['experiment_cfg']))[expt_name.lower()]
            numbins = expt_cfg['numbins']
            # print (numbins)
            numbins = 200
            a_num = expt_cfg['acquisition_num']
            ns = expt_cfg['num_seq_sets']
            readout_length = quantum_device_cfg['readout']['length']
            window = quantum_device_cfg['readout']['window']
            atten = quantum_device_cfg['readout']['dig_atten']
            # print ('Readout length = ',readout_length)
            # print ('Readout window = ',window)
            # print ("Digital atten = ",atten)
            # print ("Readout freq = ",quantum_device_cfg['readout']['freq'])
            I = np.array(a["I"])
            Q = np.array(a["Q"])
            sample = a_num
            # print(np.shape(I))
            I, Q = pd.DataFrame(I/2**15*ran), pd.DataFrame(Q/2**15*ran)
        #ENDWITH
    #ENDFOR
    ig, qg = I.iloc[0], Q.iloc[0]
    ie, qe = I.iloc[1], Q.iloc[1]
    if_, qf = I.iloc[2], Q.iloc[2]
    g_samples = np.array(list(zip(ig, qg)))
    e_samples = np.array(list(zip(ie, qe)))
    f_samples = np.array(list(zip(if_, qf)))
    return (g_samples, e_samples, f_samples)
#ENDDEF


"""
slice_for_validation - split iq_data into a training set and a validation
set
"""
def slice_for_validation(iq_data, slice_index, slice_count=VALIDATION_COUNT):
    len_ = iq_data.shape[0]
    slice_len = int(np.floor(len_ / slice_count))
    indlo = slice_len * slice_index
    indhi = slice_len * (slice_index + 1)
    train = np.vstack([iq_data[:indlo, :], iq_data[indhi:, :]])
    validate = iq_data[indlo:indhi, :]
    return (train, validate)
#ENDDEF


"""
find_nearest_index_2d - given a 2d point, find the index of the nearest point
in an array of 2d points
"""
def find_nearest_index_2d(point, array):
    dist = np.zeros(array.shape[0])
    for i in range(array.shape[0]):
        dist[i] = np.sqrt((point[0] - array[i, 0]) ** 2 + (point[1] - array[i, 1]) ** 2)
        # print("dist[{}]: {}"
        #       "".format(i, dist[i]))
    #ENDFOR
    return np.argmin(dist)
#ENDDEF


def plot_gmm(gmm, samples=None, colors=None, labels=None, gmm_colors=None,
             ylim=None, xlim=None, plot_file_path=PLOT_FILE_PATH, dpi=DPI_FINAL):
    """
    plot_gmm - plot the gaussians associated with a gaussian mixture model and some data that it is fitting
    
    Arugments:
    gmm :: sklearn.mixture.GaussianMixture - already fitted
    samples :: ndarray{Float64, (N, M, O)} where N is the number of components, M is the
        number of samples, and O is the degrees of freedom per sample. Here
        O is assumed to be 2 for I, Q samples.
    colors :: list{color-like, N}
    labels :: list{string, N}
    gmm_colors :: list{color-like, N}

    Returns: absolutely nothing

    References:
    [0] https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html
        #sphx-glr-auto-examples-mixture-plot-gmm-covariances-py
    """
    data_alpha = 0.3
    data_ms = 5
    data_marker = "."
    mean_ms = 80
    mean_marker = "P"
    mean_ec = "black"
    mean_lw = 1
    ellipse_alpha = 0.1
    ellipse_ec = "black"
    ellipse_lw = 0.5
    fig = plt.figure()
    ax = plt.gca()
    gmm_params = gmm.get_params()
    component_count = gmm_params["n_components"]
    #ENDFOR
    for i in range(samples.shape[0]):
        color = None if colors is None else colors[i]
        if labels is not None:
            plt.scatter([], [], s=data_ms, color=color, label=labels[i])
        #ENDIF
        for j in range(samples.shape[1]):
            plt.scatter([samples[i, j, 0]], [samples[i, j, 1]], alpha=data_alpha,
                        s=data_ms, color=colors[i], marker=data_marker)
        #ENDFOR
    #ENDFOR
    for i in range(component_count):
        color = None if gmm_colors is None else gmm_colors[i]
        covariances = gmm.covariances_[i][:2, :2]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(
            gmm.means_[i, :2], v[0], v[1],
            180 + angle, facecolor=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(ellipse_alpha)
        ax.add_artist(ell)
        ellb = mpl.patches.Ellipse(
            gmm.means_[i, :2], v[0], v[1],
            180 + angle, facecolor="none", edgecolor=ellipse_ec,
            linewidth=ellipse_lw)
        ellb.set_clip_box(ax.bbox)
        ellb.set_alpha(1)
        ax.add_artist(ellb)
        ax.set_aspect('equal', 'datalim')
        plt.scatter([gmm.means_[i, 0]], [gmm.means_[i, 1]], marker=mean_marker, s=mean_ms,
                    color=color, edgecolors=mean_ec, linewidths=mean_lw)
    #ENDFOR
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1))
    plt.subplots_adjust(left=.15, top=1, right=1)
    plt.savefig(plot_file_path, dpi=dpi)
    plt.close(fig)
#ENDDEF


def validate_plot():
    """
    Test the efficiency of the gmm and plot it.

    Arguments: None
    Returns: None
    """
    # for reproducability
    random.seed(0)
    np.random.seed(0)
    colors = ["blue", "green", "red"]
    labels = ["g", "e", "f"]

    # grab data
    (g_samples, e_samples, f_samples) = grab_data()

    # if you are using this for an experiment you should
    # use all of your data in training like so:
    # (g_samples, e_samples, f_samples) = grab_data()
    # train_stack = np.vstack([g_samples, e_samples, f_samples])
    # gmm = GaussianMixture(COMPONENT_COUNT)
    # gmm.fit(train_stack)

    # do k-slice validation for illustrative purposes
    g_accuracy = e_accuracy = f_accuracy = 0
    for i in range(VALIDATION_COUNT):
        (g_train, g_validate) = slice_for_validation(g_samples, i)
        (e_train, e_validate) = slice_for_validation(e_samples, i)
        (f_train, f_validate) = slice_for_validation(f_samples, i)
        train_stack = np.vstack([g_train, e_train, f_train])

        # fit the gmm to data
        gmm = GaussianMixture(COMPONENT_COUNT)
        gmm.fit(train_stack)

        # determine the labels of each component by proximity to mean
        g_mean = np.mean(g_train, axis=0)
        e_mean = np.mean(e_train, axis=0)
        f_mean = np.mean(f_train, axis=0)
        g_ind = find_nearest_index_2d(g_mean, gmm.means_)
        e_ind = find_nearest_index_2d(e_mean, gmm.means_)
        f_ind = find_nearest_index_2d(f_mean, gmm.means_)
        # print("means:\n{}\ngmm.means_:\n{}\ninds:\n{}"
        #       "".format(np.array([g_mean, e_mean, f_mean]), gmm.means_,
        #                 np.array([g_ind, e_ind, f_ind])))

        # validate
        g_predicted = gmm.predict(g_validate)
        e_predicted = gmm.predict(e_validate)
        f_predicted = gmm.predict(f_validate)
        g_accuracy = g_accuracy + np.count_nonzero(g_predicted == g_ind) / g_validate.shape[0]
        e_accuracy = e_accuracy + np.count_nonzero(e_predicted == e_ind) / e_validate.shape[0]
        f_accuracy = f_accuracy + np.count_nonzero(f_predicted == f_ind) / f_validate.shape[0]
    #ENDFOR
    # average the accuracies
    g_accuracy = g_accuracy / VALIDATION_COUNT
    e_accuracy = e_accuracy / VALIDATION_COUNT
    f_accuracy = f_accuracy / VALIDATION_COUNT

    # plot the gmm and data
    gmm_colors = [colors[g_ind], colors[e_ind], colors[f_ind]]
    samples_sep = np.array([g_samples, e_samples, f_samples])
    plot_gmm(gmm, samples=samples_sep,
             colors=colors, labels=labels, gmm_colors=gmm_colors,
             ylim=(-0.025, -0.015), xlim=(-0.01, 0.005))
    print("g_acc: {}, e_acc: {}, f_acc: {}"
          "".format(g_accuracy, e_accuracy, f_accuracy))
#ENDDEF


def make_gif(max_iter=MAX_ITER, seed=0):
    """
    this doesn't work currently because the random initial guess diverges
    and the kmeans initialization is too good
    """
    colors = ["blue", "green", "red"]
    labels = ["g", "e", "f"]

    # grab data
    (g_samples, e_samples, f_samples) = iq_data = grab_data()
    train_stack = np.vstack([g_samples, e_samples, f_samples])
    samples_sep = np.array([g_samples, e_samples, f_samples])
    g_mean = np.mean(g_samples, axis=0)
    e_mean = np.mean(e_samples, axis=0)
    f_mean = np.mean(f_samples, axis=0)
    g_ind = 0
    e_ind = 1
    f_ind = 2
    precisions_init = np.array([
        [[1e-6, 1e-7],
         [1e-7, 1e-6],],
        [[1e-6, 1e-7],
         [1e-7, 1e-6],],
        np.linalg.inv([[2.32531177e-6, 3.05499850e-7],
                       [3.05499850e-7, 2.83755594e-6],]),

    ])
    
    for i in range(1, max_iter + 1):
        random.seed(0)
        np.random.seed(0)

        plot_file_path = os.path.join(SAVE_PATH, "{:05d}_iqp.png".format(i))
        gmm = GaussianMixture(COMPONENT_COUNT, max_iter=i, init_params="random",
                              means_init=np.array([g_mean, e_mean, f_mean]),
                              weights_init=np.ones(COMPONENT_COUNT) / COMPONENT_COUNT,
                              precisions_init=precisions_init,
                              random_state=seed,
        )
        gmm.fit(train_stack)

        # g_ind = find_nearest_index_2d(g_mean, gmm.means_)
        # e_ind = find_nearest_index_2d(e_mean, gmm.means_)
        # f_ind = find_nearest_index_2d(f_mean, gmm.means_)
        gmm_colors = [colors[g_ind], colors[e_ind], colors[f_ind]]
        
        plot_gmm(gmm, samples=samples_sep, colors=colors, gmm_colors=gmm_colors,
                 labels=labels, ylim=(-0.025, -0.015), xlim=(-0.01, 0.005),
                 plot_file_path=plot_file_path, dpi=DPI)
        print("pfp: {}".format(plot_file_path))
    #ENDDEF
#ENDDEF


def main():
    validate_plot()
#ENDDEF


if __name__ == "__main__":
    main()
#ENDIF

