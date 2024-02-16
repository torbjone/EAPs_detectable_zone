import numpy as np
import os
from os.path import join
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from plotting_convention import mark_subplots, simplify_axes
import pandas
from sklearn.decomposition import PCA
from sklearn import manifold

root_folder = os.path.abspath(join(os.path.dirname(__file__), '..'))


exp_data_folder = join(root_folder, "exp_data", "NPUltraWaveforms")
sim_data_folder = join(root_folder, "exp_data", "simulated")

fig_folder = join(exp_data_folder, "figures")
os.makedirs(fig_folder, exist_ok=True)


x = np.load(join(exp_data_folder, "channels.xcoords.npy"))[:, 0]
z = np.load(join(exp_data_folder, "channels.ycoords.npy"))[:, 0]

spike_times = np.load(join(exp_data_folder, "spikes.times.npy"))
spike_clusters = np.load(join(exp_data_folder, "spikes.clusters.npy"))
waveforms = np.load(join(exp_data_folder, "clusters.waveforms.npy"))
meta_data = pandas.read_csv(join(exp_data_folder, "clusters.acronym.tsv"), sep='\t')
depth_sort = np.argsort(z)

# print(waveforms.shape)
dx = 6
dz = 6

detection_threshold = 30  # µV

num_elecs = len(x)
exp_num_tsteps = waveforms.shape[1]
num_neurons = waveforms.shape[0]
sampling_rate = 30000  # Hz
exp_dt = 1 / sampling_rate * 1000
#print(dt)
grid_shape = (int(num_elecs / 8), 8)

x_grid = x.reshape(grid_shape)
z_grid = z.reshape(grid_shape)

#tstop = 2
exp_tvec = np.arange(exp_num_tsteps) * exp_dt

#print(exp_tvec)
#print("Num elecs: ", num_elecs)

#print(x.shape)
#print(spike_times.shape)
#print(spike_clusters.shape)
print(waveforms.shape)

def plot_NPUltraWaveform(waveform, tvec, fig_name, fig_folder, cell=None,
                         acronym=None, level=None):

    xlim = [-10, np.max(np.abs(x)) + 10]
    ylim = [-10, np.max(np.abs(z)) + 10]

    plt.close("all")
    fig = plt.figure(figsize=[16, 9])
    ax1 = fig.add_axes([0.01, 0.01, 0.12, 0.98], aspect=1, xlim=xlim, ylim=ylim,
                       xticks=[], yticks=[], frameon=False)
    ax2 = fig.add_axes([0.17, 0.01, 0.12, 0.98], aspect=1, xlim=xlim, ylim=ylim,
                       xticks=[], yticks=[], frameon=False,)
    ax3 = fig.add_axes([0.32, 0.01, 0.22, 0.95], ylabel="electrode number",
                       xticks=[], yticks=[], frameon=False)
    ax4 = fig.add_axes([0.63, 0.1, 0.35, 0.86], xlabel="time (ms)",
                       ylabel="µV")

    if acronym is not None:
        fig.text(0.64, 0.93, acronym)
    if level is not None:
        fig.text(0.64, 0.96, level)

    #cax_fp = fig.add_axes([0.28, 0.1, 0.01, 0.2])
    cax_t = fig.add_axes([0.55, 0.2, 0.01, 0.2])

    ax1.scatter(x, z, c='gray', s=1)
    if cell is not None:

        zips = []
        for x1, x2 in cell.get_idx_polygons(projection="xz"):
            zips.append(list(zip(x1, x2)))
        polycol = PolyCollection(zips,
                                 edgecolors='none',
                                 facecolors='0.8',
                                 rasterized=True)

        ax1.add_collection(polycol)
        from main import plot_superimposed_sec_type
        plot_superimposed_sec_type(cell, ax1)

    eap_norm = np.max(np.abs(waveform))
    max_peak_idxs = np.argmax(np.abs(waveform), axis=0)
    neg_peak = np.min(waveform, axis=0)
    max_elec_peak = np.argmax(np.max(np.abs(waveform), axis=0))
    max_t_idx = np.argmax(np.abs(waveform[:, max_elec_peak]))

    img = ax3.imshow(waveform[:, depth_sort].T, cmap="bwr",
               vmax=eap_norm, vmin=-eap_norm, origin="lower",
               extent=[0, tvec[-1], 0, np.max(z)])
    ax3.axis("auto")
    cbar = plt.colorbar(img, cax=cax_t)
    cbar.set_label("µV")

    ax3.plot([1, 2], [-2, -2], c='k')
    ax3.text(1.5, -3, "1 ms", va="top", ha="center")
    amp_peak = np.array([waveform[idx_, i_] for i_, idx_ in enumerate(max_peak_idxs)])

    img2 = ax2.contourf(x_grid, z_grid, amp_peak.reshape(grid_shape), levels=25,
                    cmap="bwr", vmin=-eap_norm, vmax=eap_norm)

    levels = (-1e9, -detection_threshold, 1e9)
    ax2.contour(x_grid, z_grid, neg_peak.reshape(grid_shape),
                levels=levels, colors='cyan')

    #cbar2 = plt.colorbar(img2, cax=cax_fp)
    #cbar2.set_ticks([-eap_norm, 0, eap_norm])
    #cax_fp.axhline(-detection_threshold, c='cyan', ls='--')
    #cbar2.set_label("µV")
    x_norm = 0.8 * dx / tvec[-1]
    y_norm = 0.9 * dz / eap_norm
    eap_clr = lambda eap_min: plt.cm.viridis(0.0 + eap_min / np.min(neg_peak))
    for elec_idx in range(num_elecs):
        x_ = x[elec_idx] + tvec * x_norm
        y_ = z[elec_idx] + waveform[:, elec_idx] * y_norm
        ax1.plot(x_, y_, lw=1, c='k', zorder=10)
        ax4.plot(tvec, waveform[:, elec_idx], zorder=-neg_peak[elec_idx],
                 c=eap_clr(neg_peak[elec_idx]))

    ax4.axhline(-detection_threshold, c='cyan', ls="--")
    ax4.text(0.1, -detection_threshold*1.05,
             "detection\nthreshold", c="cyan", va="top")
    ax4.axvline(tvec[max_t_idx], lw=0.5, c='gray', ls=":")
    ax4.text(tvec[max_t_idx] + 0.1, waveform[max_t_idx, max_elec_peak], tvec[max_t_idx])

    ax1.plot([50, 50], [200, 200 + eap_norm * y_norm], c='k')
    ax1.text(52, 200 + eap_norm * y_norm / 2, "{:1.1f} µV".format(eap_norm),
             va="center", ha="left")

    ax1.plot([50, 50], [30 * dz, 30 * dz + dz], c='gray')
    ax1.text(52, 30 * dz + dz / 2, "{:d} µm".format(dz),
             va="center", ha="left", color="gray")

    ax2.plot([45, 45], [30 * dz, 30 * dz + dz], c='gray')
    ax2.text(47, 30 * dz + dz / 2, "{:d} µm".format(dz),
             va="center", ha="left", color="gray")

    ax1.plot(x[max_elec_peak], z[max_elec_peak], 'ro')
    ax1.plot([1 * dx, 1 * dx + dx], [-2, -2], c='gray')
    ax1.text(1 * dx + dx / 2, -3, "{:d} µm".format(dx),
             va="top", ha="center", color='gray')

    ax1.plot([6 * dx, 6 * dx + tvec[-1] * x_norm], [-2, -2], c='k')
    ax1.text(6 * dx + tvec[-1] * x_norm / 2, -3, "{:1.1f} ms".format(tvec[-1]),
             va="top", ha="center")

    simplify_axes(ax4)
    mark_subplots([ax1, ax2], xpos=0.05, ypos=1.0)
    mark_subplots([ax3, ax4], ["C", "D"], xpos=0.0, ypos=1.03)
    fig.savefig(join(fig_folder, "waveform_%s.png" % fig_name), dpi=150)


def analyze_elec_size_SNR():
    spike_idx = 8
    waveform = waveforms[spike_idx]
    tvec = exp_tvec

    eap_norm = np.max(np.abs(waveform))
    max_peak_idxs = np.argmax(np.abs(waveform), axis=0)
    neg_peak = np.min(waveform, axis=0)
    max_elec_peak = np.argmax(np.max(np.abs(waveform), axis=0))
    max_t_idx = np.argmax(np.abs(waveform[:, max_elec_peak]))

    elec_dist_from_max = np.sqrt((x - x[max_elec_peak])**2 + (z - z[max_elec_peak])**2)

    print(waveform.shape)

    xlim = [-10, np.max(np.abs(x)) + 10]
    ylim = [-10, np.max(np.abs(z)) + 10]

    plt.close("all")
    fig = plt.figure(figsize=[16, 9])
    ax1 = fig.add_axes([0.01, 0.01, 0.12, 0.98], aspect=1, xlim=xlim, ylim=ylim,
                       xticks=[], yticks=[], frameon=False)
    ax_snr = fig.add_axes([0.15, 0.8, 0.12, 0.15],)

    ax4 = fig.add_axes([0.63, 0.1, 0.35, 0.86], xlabel="time (ms)",
                       ylabel="µV")

    ax_snr.plot(elec_dist_from_max, np.std(waveform, axis=0))

    ax1.scatter(x, z, c='gray', s=1)

    x_norm = 0.8 * dx / tvec[-1]
    y_norm = 0.9 * dz / eap_norm
    eap_clr = lambda eap_min: plt.cm.viridis(0.0 + eap_min / np.min(neg_peak))
    y_norm = 0.9 * dz / eap_norm
    eap_clr = lambda eap_min: plt.cm.viridis(0.0 + eap_min / np.min(neg_peak))
    for elec_idx in range(num_elecs):
        x_ = x[elec_idx] + tvec * x_norm
        y_ = z[elec_idx] + waveform[:, elec_idx] * y_norm
        ax1.plot(x_, y_, lw=1, c='k', zorder=10)
        ax4.plot(tvec, waveform[:, elec_idx], zorder=-neg_peak[elec_idx],
                 c=eap_clr(neg_peak[elec_idx]))

    plt.savefig(f"elec_size_SNR_idx:{spike_idx}.png")

def extract_spike_features(waveform, tvec, fig_name, fig_folder,
                           plot_it=True, soma_dist_from_plane=None):
    dt = tvec[1] - tvec[0]

    eap_norm = np.max(np.abs(waveform))
    # max_peak_idxs = np.argmax(np.abs(waveform), axis=0)
    p2ps = np.max(waveform, axis=0) - np.min(waveform, axis=0)
    neg_peak = np.min(waveform, axis=0)
    largest_eap_idx = np.argmax(p2ps)
    largest_eap = waveform[:, largest_eap_idx]
    max_peak_tidx = np.argmax(np.abs(largest_eap))
    peak_tidx = waveform[max_peak_tidx, :]

    max_peak = largest_eap[max_peak_tidx]
    if max_peak_tidx == 0:
        print("RRORORROR")
        print(fig_name)
        prepeak_amp = 0
        # return None

    else:
        #print(tvec[max_neg_peak_tidx])
        # print(len(tvec), max_neg_peak_tidx)
        if max_peak < 0:
            prepeak_amp = np.max(largest_eap[:max_peak_tidx])
        else:
            prepeak_amp = np.min(largest_eap[:max_peak_tidx])

    max_p2p = np.max(p2ps)
    if max_peak < 0:
        tidx_max_postpeak = np.argmax(largest_eap[max_peak_tidx:])
    else:
        tidx_max_postpeak = np.argmin(largest_eap[max_peak_tidx:])

    postpeak_delay = tidx_max_postpeak * dt

    if max_peak < 0:
        fwhm = return_spike_width(largest_eap, dt)
    else:
        fwhm = return_spike_width(-largest_eap, dt)

    # neg_peak_tidx = waveform[max_peak_tidx, :]
    max_idx = largest_eap_idx#np.argmin(neg_peak_tidx)
    #max_idx_p2p = np.argmax(p2ps)

    large_amp_idxs = np.where(np.abs(peak_tidx) >= 20)
    from scipy.optimize import minimize

    #loc_min_func = lambda args: np.sum(np.abs(args[3] * np.abs(args[1])/np.sqrt(
    #      (x[large_amp_idxs] - args[0]) ** 2
    #    + (args[1]) ** 2
    #    + (z[large_amp_idxs] - args[2]) ** 2
    #    ) - neg_peak_tidx[large_amp_idxs]))
    #res = minimize(loc_min_func, (x[max_idx], 20., z[max_idx], max_neg_peak),
    #               method='SLSQP', bounds=((x[max_idx] - 2*dx, x[max_idx] + 2*dx),
    #                                       (0., 100.),
    #                                       (z[max_idx] - 2*dz, z[max_idx] + 2*dz),
    #                                       (max_neg_peak*10, max_neg_peak)
    #                                       ))
    # print(res.x)
    # fit_xloc = res.x[0]
    # fit_height = res.x[1]
    # fit_zloc = res.x[2]

    dist_from_max = np.sqrt((x - x[max_idx]) ** 2 + (z - z[max_idx]) ** 2)
    # decay_func = lambda h_: max_neg_peak / np.sqrt(1 + (dist_from_max[large_amp_idxs] / h_)**2)

    decay_func = lambda lambd: max_peak * np.exp(-dist_from_max[large_amp_idxs] / lambd)

    minimize_func = lambda lambd: np.sum(np.abs((decay_func(lambd) -
                                      peak_tidx[large_amp_idxs]) / peak_tidx[large_amp_idxs]))


    res = minimize(minimize_func, (20,), method='SLSQP', bounds=((1, 200.), ))
    fit_decay = res.x[0]
    decay_fit = decay_func(fit_decay)
    # print(fit_decay)
    if plot_it:
        xlim = [-10, np.max(np.abs(x)) + 10]
        ylim = [-10, np.max(np.abs(z)) + 10]

        plt.close("all")
        fig = plt.figure(figsize=[16, 9])
        ax1 = fig.add_axes([0.00, 0.01, 0.11, 0.98], aspect=1, xlim=xlim, ylim=ylim,
                           xticks=[], yticks=[], frameon=False)
        ax_p2p = fig.add_axes([0.12, 0.01, 0.11, 0.98], aspect=1, xlim=xlim, ylim=ylim,
                           xticks=[], yticks=[], frameon=False)
        ax_negpeak = fig.add_axes([0.24, 0.01, 0.11, 0.98], aspect=1, xlim=xlim, ylim=ylim,
                           xticks=[], yticks=[], frameon=False)
        ax_negpeak_tidx = fig.add_axes([0.36, 0.01, 0.11, 0.98], aspect=1, xlim=xlim, ylim=ylim,
                           xticks=[], yticks=[], frameon=False)
        ax_amp_decay = fig.add_axes([0.5, 0.6, 0.1, 0.3], xlim=[0, 50], ylim=[1, 1000])

        ax_all_eaps = fig.add_axes([0.60, 0.52, 0.35, 0.48],
                           frameon=False, xticks=[], yticks=[])
        ax_largest_eap = fig.add_axes([0.60, 0.02, 0.35, 0.48], frameon=False,
                           xticks=[], yticks=[])

        ax_amp_decay.semilogy(dist_from_max, np.abs(waveform[max_peak_tidx, :]), 'k.')
        l_, = ax_amp_decay.semilogy(dist_from_max[large_amp_idxs], np.abs(decay_fit), 'r.')
        l_name = r"$A \exp^{-r / \lambda}$"
        l_name += "; $\lambda$={:1.1f} µm".format(fit_decay)
        #l_name = "point source\nh={:1.1f} µm".format(fit_height)
        #if soma_dist_from_plane is not None:
        #    l_name += '\ntrue: {:1.1f} µm'.format(-soma_dist_from_plane)
        ax_amp_decay.legend([l_], [l_name], frameon=False,
                           loc=(0, 1.05))

        #ax1.plot(fit_xloc, fit_zloc, 'rx')
        #ax_p2p.plot(fit_xloc, fit_zloc, 'rx')
        #ax_negpeak.plot(fit_xloc, fit_zloc, 'rx')
        #ax_negpeak_tidx.plot(fit_xloc, fit_zloc, 'rx')

        x_norm = 0.8 * dx / tvec[-1]
        y_norm = 0.9 * dz / eap_norm
        eap_clr = lambda eap_min: plt.cm.viridis(0.0 + eap_min / np.min(neg_peak))
        for elec_idx in range(num_elecs):
            x_ = x[elec_idx] + tvec * x_norm
            y_ = z[elec_idx] + waveform[:, elec_idx] * y_norm
            ax1.plot(x_, y_, lw=1, c='k', zorder=10)
            ax_all_eaps.plot(tvec, waveform[:, elec_idx], zorder=-neg_peak[elec_idx],
                     c=eap_clr(neg_peak[elec_idx]))

        img2 = ax_p2p.contourf(x_grid, z_grid, p2ps.reshape(grid_shape), levels=25,
                            cmap="bwr", vmin=-eap_norm, vmax=eap_norm)

        ax_negpeak.contourf(x_grid, z_grid, neg_peak.reshape(grid_shape), levels=25,
                        cmap="bwr", vmin=-eap_norm, vmax=eap_norm)
        ax_negpeak_tidx.contourf(x_grid, z_grid, waveform[max_peak_tidx, :].reshape(grid_shape), levels=25,
                        cmap="bwr", vmin=-eap_norm, vmax=eap_norm)

        ax_largest_eap.plot(tvec, largest_eap, 'k', lw=3)
        ax_largest_eap.plot([1.3, 1.3 + fwhm], [np.min(largest_eap) / 2, np.min(largest_eap) / 2], c='r', lw=2)
        ax_largest_eap.text(1.4 + fwhm, np.min(largest_eap) / 2, "{:1.2f}\nms".format(fwhm), va="center", ha="left", color='r')

        ax_largest_eap.plot([tvec[max_peak_tidx], tvec[max_peak_tidx] + postpeak_delay],
                 [np.max(largest_eap) / 2, np.max(largest_eap) / 2], c='b', lw=2)
        ax_largest_eap.text(tvec[max_peak_tidx], np.max(largest_eap)/1.5, "{:1.2f}\nms".format(postpeak_delay),
                 va="center", ha="left", color='b')

        ax_largest_eap.plot([1, 1], [0, max_peak], c='green', lw=2)
        ax_largest_eap.text(0.9, max_peak/2, "{:1.1f}\nµV".format(max_peak), va="center", ha="right", color='green')

        ax_largest_eap.plot([tvec[-1], tvec[-1]], [max_peak, max_peak + max_p2p], c='orange', lw=2)
        ax_largest_eap.text(tvec[-1] + 0.1, (max_p2p + max_peak)/2, "{:1.1f}\nµV".format(max_p2p), va="center", ha="left", color='orange')

        ax_largest_eap.plot([0.75, 0.75], [0, prepeak_amp], c='purple', lw=2)
        ax_largest_eap.text(0.7, prepeak_amp/1.5, "{:1.1f}\nµV".format(prepeak_amp), va="center", ha="right", color='purple')

        simplify_axes(ax_amp_decay)
        fig.savefig(join(fig_folder, "%s.png" % fig_name))

    if np.isnan(fwhm):
        print(fwhm)
        print(fig_name)
        raise RuntimeError("fwhm NaN!")

    eap_feature_dict = {
        'p2p': max_p2p,
        'max_peak': max_peak,
        'prepeak': prepeak_amp,
        'fwhm': fwhm,
        'postpeak_delay': postpeak_delay,
        'decay': fit_decay,
    }
    return eap_feature_dict


def animate_NPUltraWaveform(waveform, tvec, fig_name, fig_folder, cell=None):

    xlim = [-10, np.max(np.abs(x)) + 10]
    ylim = [-10, np.max(np.abs(z)) + 10]

    plt.close("all")
    fig = plt.figure(figsize=[16, 9])
    ax1 = fig.add_axes([0.01, 0.01, 0.12, 0.98], aspect=1, xlim=xlim, ylim=ylim,
                       xticks=[], yticks=[], frameon=False)
    ax2 = fig.add_axes([0.17, 0.01, 0.12, 0.98], aspect=1, xlim=xlim, ylim=ylim,
                       xticks=[], yticks=[], frameon=False,)
    ax3 = fig.add_axes([0.32, 0.01, 0.22, 0.95], ylabel="electrode number",
                       xticks=[], yticks=[], frameon=False)
    ax4 = fig.add_axes([0.63, 0.1, 0.35, 0.86], xlabel="time (ms)",
                       ylabel="µV")

    #cax_fp = fig.add_axes([0.28, 0.1, 0.01, 0.2])
    cax_t = fig.add_axes([0.55, 0.2, 0.01, 0.2])

    ax1.scatter(x, z, c='gray', s=1)
    if cell is not None:

        zips = []
        for x1, x2 in cell.get_idx_polygons(projection="xz"):
            zips.append(list(zip(x1, x2)))
        polycol = PolyCollection(zips,
                                 edgecolors='none',
                                 facecolors='0.8',
                                 rasterized=True)

        ax1.add_collection(polycol)

    eap_norm = np.max(np.abs(waveform))

    max_peak_idxs = np.argmax(np.abs(waveform), axis=0)
    neg_peak = np.min(waveform, axis=0)

    img = ax3.imshow(waveform[:, depth_sort].T, cmap="bwr_r",
               vmax=eap_norm, vmin=-eap_norm, origin="lower",
               extent=[0, tvec[-1], 0, np.max(z)])
    ax3.axis("auto")
    cbar = plt.colorbar(img, cax=cax_t)
    cbar.set_label("µV")

    ax3.plot([1, 2], [-2, -2], c='k')
    ax3.text(1.5, -3, "1 ms", va="top", ha="center")



    #levels = (-1e9, -detection_threshold, 1e9)
    #ax2.contour(x_grid, z_grid, neg_peak.reshape(grid_shape),
    #            levels=levels, colors='cyan')

    #cbar2 = plt.colorbar(img2, cax=cax_fp)
    #cbar2.set_ticks([-eap_norm, 0, eap_norm])
    #cax_fp.axhline(-detection_threshold, c='cyan', ls='--')
    #cbar2.set_label("µV")
    x_norm = 0.8 * dx / tvec[-1]
    y_norm = 0.9 * dz / eap_norm
    eap_clr = lambda eap_min: plt.cm.viridis(0.0 + eap_min / np.min(neg_peak))
    for elec_idx in range(num_elecs):
        x_ = x[elec_idx] + tvec * x_norm
        y_ = z[elec_idx] + waveform[:, elec_idx] * y_norm
        ax1.plot(x_, y_, lw=1, c='k')
        ax4.plot(tvec, waveform[:, elec_idx], zorder=-neg_peak[elec_idx],
                 c=eap_clr(neg_peak[elec_idx]))

    ax4.axhline(-detection_threshold, c='cyan', ls="--")
    ax4.text(0.1, -detection_threshold*1.05,
             "detection\nthreshold", c="cyan", va="top")

    ax1.plot([50, 50], [200, 200 + eap_norm * y_norm], c='k')
    ax1.text(52, 200 + eap_norm * y_norm / 2, "{:1.1f} µV".format(eap_norm),
             va="center", ha="left")

    ax1.plot([50, 50], [30 * dz, 30 * dz + dz], c='gray')
    ax1.text(52, 30 * dz + dz / 2, "{:d} µm".format(dz),
             va="center", ha="left", color="gray")

    ax2.plot([45, 45], [30 * dz, 30 * dz + dz], c='gray')
    ax2.text(47, 30 * dz + dz / 2, "{:d} µm".format(dz),
             va="center", ha="left", color="gray")

    ax1.plot([1 * dx, 1 * dx + dx], [-2, -2], c='gray')
    ax1.text(1 * dx + dx / 2, -3, "{:d} µm".format(dx),
             va="top", ha="center", color='gray')

    ax1.plot([6 * dx, 6 * dx + tvec[-1] * x_norm], [-2, -2], c='k')
    ax1.text(6 * dx + tvec[-1] * x_norm / 2, -3, "{:1.1f} ms".format(tvec[-1]),
             va="top", ha="center")

    simplify_axes(ax4)
    mark_subplots([ax1, ax2], xpos=0.05, ypos=1.0)
    mark_subplots([ax3, ax4], ["C", "D"], xpos=0.0, ypos=1.03)
    os.makedirs(fig_folder, exist_ok=True)

    t_idx = 0
    lax3 = ax3.axvline(tvec[t_idx], ls='--', c='gray')
    lax4 = ax4.axvline(tvec[t_idx], ls='--', c='gray')
    time_marker = ax2.text(1, ylim[1] - 2.2, "t = {:1.2f} ms".format(tvec[t_idx]))
    img2 = ax2.contourf(x_grid, z_grid, waveform[t_idx, :].reshape(grid_shape), levels=25,
                    cmap="bwr_r", vmin=-eap_norm, vmax=eap_norm)

    for t_idx in range(len(tvec)):
        print(t_idx + 1, len(tvec))
        lax3.set_xdata(tvec[t_idx])
        lax4.set_xdata(tvec[t_idx])
        for coll in img2.collections:
            coll.remove()

        img2 = ax2.contourf(x_grid, z_grid, waveform[t_idx, :].reshape(grid_shape), levels=25,
                                cmap="bwr_r", vmin=-eap_norm, vmax=eap_norm)
        time_marker.set_text("t = {:1.2f} ms".format(tvec[t_idx]))
        fig.savefig(join(fig_folder, "waveform_%s_%04d.png" % (fig_name, t_idx)), dpi=150)

    cmd = "ffmpeg -framerate 5 -i %s" % join(fig_folder, "waveform_{:s}_%04d.png ".format(fig_name))
    cmd += "-c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p "
    cmd += "%s.avi" % join(fig_folder, "..", "..", fig_name)
    # print(cmd)
    os.system(cmd)
    rm_cmd = "rm %s/*.png" % fig_folder
    # print(rm_cmd)
    os.system(rm_cmd)


def return_spike_width(eap, dt):
    """ Full width half maximum"""

    if np.max(eap) > np.abs(np.min(eap)):
        return np.NaN

    half_max = np.min(eap) / 2
    t0_idx = None
    t1_idx = None
    for t_idx in range(1, len(eap)):
        if eap[t_idx - 1] > half_max >= eap[t_idx]:
            t0_idx = t_idx if t0_idx is None else t0_idx  # Only record 1st crossing if more
        if eap[t_idx - 1] < half_max <= eap[t_idx]:
            t1_idx = t_idx if t1_idx is None else t1_idx  # Only record 1st crossing if more
    if t0_idx is None:
        print("t0 is None!")
        t0_idx = 0
        # plt.plot(eap)
        # plt.show()
        #return np.NaN
    if t1_idx is None:
        print("t1 is None!")
        t1_idx = len(eap)
        # plt.plot(eap)
        # plt.show()
        # return np.NaN
    return (t1_idx - t0_idx) * dt


def analyse_waveform_collection(waveforms, tvec, name, celltypes_list=None):

    max_amps = []
    max_amp_xz = []
    max_amp_width = []
    num_elecs_above_threshold = []
    dt = tvec[1] - tvec[0]

    color_list = []
    acronym_subtype = ""#"VISp5"
    for neuron_id in range(waveforms.shape[0]):

        # acronym = meta_data.acronym[neuron_id]
        # level = meta_data.level6[neuron_id]

        # if not acronym == acronym_subtype:
        #     print(f"Skipping {neuron_id} because it is {acronym}")
        #     continue

        waveform = waveforms[neuron_id]
        #max_neg_peak = np.min(waveform, axis=0)
        if not celltypes_list is None:
            #print(celltypes_list[neuron_id])
            cell_type = celltypes_list[neuron_id]
            if ("aspiny") in cell_type:
                color_list.append("b")
            elif ("spiny" in cell_type) or ("PC" in cell_type) or ("SS" in cell_type) or ("SP" in cell_type):
                color_list.append("r")
            else:
                print(cell_type, " counted as inhibitory")
                color_list.append("b")
        else:
            color_list.append('k')
        p2ps = np.max(waveform, axis=0) - np.min(waveform, axis=0)
        max_amp = np.max(np.abs(waveform), axis=0)

        max_amp_idx = np.argmax(p2ps)
        #if np.max(p2ps) > 1000:
        #    plt.plot(waveform[:, max_amp_idx])
        #    plt.show()
        width = return_spike_width(waveform[:, max_amp_idx], dt)

        max_amp_xz.append([x[max_amp_idx], z[max_amp_idx]])
        max_amp_width.append(width)
        max_amps.append(np.max(p2ps))
        num_detectable_elecs = len(np.where(max_amp > detection_threshold)[0])
        # if num_detectable_elecs > 380:
        #     print(num_detectable_elecs)
        #     print(p2ps, max_amp)
        #print(num_detectable_elecs)
        num_elecs_above_threshold.append(num_detectable_elecs)

    plt.close("all")
    num_cols = 5
    num_rows = 1
    fig = plt.figure(figsize=[16, 3])
    fig.subplots_adjust(bottom=0.2, right=0.98, left=0.09)
    fig.text(0.01, 0.5, name, rotation=90, va="center")
    ax_amp = fig.add_subplot(num_rows, num_cols, 1, ylabel="#",
                             xlabel="max p2p amp (µV)",
                             xlim=[0, 1000])
    ax_width = fig.add_subplot(num_rows, num_cols, 2, ylabel="#",
                             xlabel="full width half max (ms)",
                             xlim=[0, 0.6])
    ax_detect = fig.add_subplot(num_rows, num_cols, 3, ylabel="#",
                             xlabel="# elecs > %d µV" % detection_threshold,
                             xlim=[0, 300])
    ax_amp_vs_width = fig.add_subplot(num_rows, num_cols, 4, xlabel="max p2p (µV)",
                             ylabel="width (ms)", ylim=[0, 0.6], xlim=[0, 1000])
    ax_amp_vs_detect = fig.add_subplot(num_rows, num_cols, 5,
                                       xlabel="max p2p amp (µV)",
                            ylabel="# elecs > %d µV" % detection_threshold,
                                       xlim=[0, 1000], ylim=[0, 300])

    ax_amp.hist(max_amps, bins=50, color='k')
    ax_width.hist(max_amp_width, bins=40, color='k')
    ax_detect.hist(num_elecs_above_threshold, bins=50, color='k')

    ax_amp_vs_width.scatter(max_amps, max_amp_width, c=color_list, s=3)
    ax_amp_vs_detect.scatter(max_amps, num_elecs_above_threshold, c=color_list, s=3)
    simplify_axes(fig.axes)
    plt.savefig("analysis_waveforms_%s_%s.png" % (name, acronym_subtype))
    plt.savefig("analysis_waveforms_%s_%s.pdf" % (name, acronym_subtype), dpi=100)


def analyse_simulated_waveform_collections():
    sim_dt = exp_dt
    sim_names = ["hallermann", "allen", "hay", "BBP"][:]
    for sim_name in sim_names:
        print(sim_name)
        sim_waveforms = np.load(join(sim_data_folder, "waveforms_sim_%s.npy" % sim_name))
        try:
            sim_waveforms_celltypes = np.load(join(sim_data_folder, "waveforms_sim_%s_celltype_list.npy" % sim_name))
        except FileNotFoundError:
            sim_waveforms_celltypes = None
        print(sim_waveforms.shape)
        sim_num_tsteps = sim_waveforms.shape[1]
        sim_tvec = np.arange(sim_num_tsteps) * sim_dt
        analyse_waveform_collection(sim_waveforms, sim_tvec, "sim_%s_data_" % sim_name, sim_waveforms_celltypes)


def plot_all_waveforms():
    fig_folder = join(exp_data_folder, "figures_VISp")
    os.makedirs(fig_folder, exist_ok=True)
    for neuron_id in range(num_neurons):
        print(neuron_id, "/", num_neurons)
        fig_name = "exp_data_%d" % neuron_id
        waveform = waveforms[neuron_id]
        acronym = meta_data.acronym[neuron_id]
        level = meta_data.level6[neuron_id]
        level7 = meta_data.level7[neuron_id]
        if 'VISp' in acronym:
            plot_NPUltraWaveform(waveform, exp_tvec, fig_name, fig_folder, acronym=acronym, level=level)


def animate_sim_waveform():
    sampling_rate = 30000  # Hz
    sim_dt = 1 / sampling_rate * 1000  # ???

    sim_name = ["hallermann", "allen", "hay", "BBP"][-1]
    waveform_idx = 15771
    sim_waveforms = np.load(join(sim_data_folder, "waveforms_sim_%s.npy" % sim_name))
    sim_num_tsteps = sim_waveforms.shape[1]
    sim_tvec = np.arange(sim_num_tsteps) * sim_dt
    fig_folder_sim = join(sim_data_folder, "anim", sim_name)
    animate_NPUltraWaveform(sim_waveforms[waveform_idx],
                            sim_tvec, "anim_%s_%d" % (sim_name, waveform_idx),
                            join(fig_folder_sim))


def single_waveform_pca(waveform, tvec, fig_name, fig_folder):
    n_components = 1
    eap_norm = np.max(np.abs(waveform))
    pca = PCA(n_components)
    projected = pca.fit_transform(waveform.T)

    approx = pca.inverse_transform(projected).T
    # print(np.sum(pca.components_[0, :]), np.std(pca.components_[0, :]), np.cov(pca.components_[0, :]))
    #approx = np.zeros_like(waveform.T[:, :]) + pca.mean_[None, :]
    #for i in range(n_components):
    #    approx += np.dot(projected[:, i, None], pca.components_[None, i, :])

    temporal_waveform = pca.components_[0, :]
    spatial_waveform = projected[:, 0]

    #temporal_shapes.append(temporal_waveform)
    #spatial_shapes.append(spatial_waveform)

    # pca_l2 = PCA(2)
    # projected_l2 = pca.fit_transform()
    eap_norm = np.max(np.abs(waveform))
    neg_peak = np.min(waveform, axis=0)
    p2p = np.max(waveform, axis=0) - np.min(waveform, axis=0)
    max_idx = np.argmax(np.max(np.abs(waveform), axis=0))
    max_idx_p2p = np.argmax(p2p)

    dist_from_max = np.sqrt((x - x[max_idx]) ** 2 + (z - z[max_idx]) ** 2)

    max_error = np.max(np.abs((waveform - approx)), axis=0)
    max_error_idx = np.argmax(max_error)

    xlim = [-2, np.max(np.abs(x)) + 2]
    ylim = [-2, np.max(np.abs(z)) + 2]
    plt.close("all")
    fig = plt.figure(figsize=[10, 10])

    ax1 = fig.add_axes([0.00, 0.01, 0.2, 0.9], aspect=1, xlim=xlim, ylim=ylim,
                       xticks=[], yticks=[], frameon=False)
    ax2 = fig.add_axes([0.03, 0.92, 0.13, 0.05], ylim=[-0.6, 0.2],
                       xticks=[], yticks=[], frameon=False)
    ax3 = fig.add_axes([0.24, 0.02, 0.2, 0.95], aspect=1, xlim=xlim, ylim=ylim,
                       xticks=[], yticks=[], frameon=False, title="Original")
    ax4 = fig.add_axes([0.42, 0.02, 0.2, 0.95], aspect=1, xlim=xlim, ylim=ylim,
                       xticks=[], yticks=[], frameon=False, title="Reconstructed")
    ax5 = fig.add_axes([0.6, 0.02, 0.2, 0.95], aspect=1, xlim=xlim, ylim=ylim,
                       xticks=[], yticks=[], frameon=False, title="Difference")
    ax6 = fig.add_axes([0.81, 0.1, 0.17, 0.12], title="largest spike",
                       xticks=[], yticks=[], frameon=False)
    ax7 = fig.add_axes([0.81, 0.24, 0.17, 0.12], title="largest error",
                       xticks=[], yticks=[], frameon=False)
    ax8 = fig.add_axes([0.83, 0.82, 0.13, 0.12], title="PCA spatial\ndecay",
                       xlim=[-10, 150], ylim=[np.max(p2p) * 1e-2, np.max(p2p) * 1.5],
                       xticks=[0, 50, 100, 150], xlabel="distance (µm)")
    ax9 = fig.add_axes([0.81, 0.53, 0.17, 0.12], title="original",
                       xticks=[], yticks=[], frameon=False)
    ax10 = fig.add_axes([0.81, 0.39, 0.17, 0.12], title="recreated",
                       xticks=[], yticks=[], frameon=False)

    ax1.set_title("spatial PCA", y=1.0)
    ax2.set_title("temporal PCA", y=1.0)
    vmax = np.max(np.abs(projected[:, 0]))

    img = ax1.contourf(x_grid, z_grid, projected[:, 0].reshape(grid_shape), levels=25,
                       cmap="bwr", vmin=-vmax, vmax=vmax)
    cax_t = fig.add_axes([0.18, 0.04, 0.01, 0.84])
    cbar = plt.colorbar(img, cax=cax_t, label="µV")

    x_norm = 0.8 * dx / tvec[-1]
    y_norm = 0.9 * dz / eap_norm
    eap_clr = lambda eap_min: plt.cm.viridis(0.0 + eap_min / np.min(neg_peak))
    for elec_idx in range(num_elecs):
        x_ = x[elec_idx] + tvec * x_norm
        y_ = z[elec_idx] + waveform[:, elec_idx] * y_norm
        y_rec = z[elec_idx] + approx[:, elec_idx] * y_norm
        y_diff = z[elec_idx] + (waveform[:, elec_idx] - approx[:, elec_idx]) * y_norm

        ax3.plot(x_, y_, lw=1, c='k', zorder=10, clip_on=False)
        ax4.plot(x_, y_rec, lw=1, c='k', zorder=10, clip_on=False)
        ax5.plot(x_, y_diff, lw=1, c='k', zorder=10, clip_on=False)

        ax9.plot(tvec, waveform[:, elec_idx], zorder=-neg_peak[elec_idx],
                 c=eap_clr(neg_peak[elec_idx]))
        ax10.plot(tvec, approx[:, elec_idx], zorder=-neg_peak[elec_idx],
                 c=eap_clr(neg_peak[elec_idx]))

    ax9.plot(tvec, -temporal_waveform / np.min(temporal_waveform) * eap_norm, 'k', lw=0.5, zorder=10000)
    ax10.plot(tvec, -temporal_waveform / np.min(temporal_waveform) * eap_norm, 'k', lw=0.5, zorder=10000)

    ax1.plot(x[max_idx], z[max_idx], 'x', c='k')

    if n_components > 1:
        for n_idx in range(1, n_components):
            ax2.plot(pca.components_[n_idx, :], c='gray', clip_on=False)
    ax2.plot(pca.components_[0, :], c='k', clip_on=False)

    l3, = ax7.plot(tvec, waveform[:, max_error_idx] - approx[:, max_error_idx], 'r')
    l1, = ax7.plot(tvec, waveform[:, max_error_idx], 'k')
    l2, = ax7.plot(tvec, approx[:, max_error_idx], 'gray')
    ax7.plot([tvec[-1] + 0.1, tvec[-1] + 0.1], [-eap_norm, 0], lw=1, c='k')
    ax7.text(tvec[-1], -eap_norm / 2, "{:1.0f}\nµV".format(eap_norm), ha='right', va="center")

    l3, = ax6.plot(tvec, waveform[:, max_idx_p2p] - approx[:, max_idx_p2p], 'r')
    l1, = ax6.plot(tvec, waveform[:, max_idx_p2p], 'k')
    l2, = ax6.plot(tvec, approx[:, max_idx_p2p], 'gray')
    fig.legend([l1, l2, l3], ["original", "PCA rec.", "difference"], ncol=1, frameon=False,
               loc=(0.8, 0.01))
    ax6.plot([tvec[-1] + 0.1, tvec[-1] + 0.1], [-eap_norm, 0], lw=1, c='k')
    ax6.text(tvec[-1], -eap_norm / 2, "{:1.0f}\nµV".format(eap_norm), ha='right', va="center")
    # ax2 = fig.add_axes([0.5, 0.2, 0.2, 0.2])
    # ax2.plot(tvec, waveform[:, max_idx], 'k')
    # ax2.plot(tvec, approx.T[:, max_idx], 'red', ls='-')

    ax8.semilogy(dist_from_max, p2p, 'k.')

    simplify_axes(ax8)
    fig.savefig(join(fig_folder, "PCA_idx_%s.png" % fig_name))
    return temporal_waveform, spatial_waveform


def analyse_pca_exp_data():
    fig_folder = join(exp_data_folder, "pca_figures")
    os.makedirs(fig_folder, exist_ok=True)


    acronym_subtype = "VISp5"
    temporal_shapes = []
    spatial_shapes = []
    for neuron_id in range(num_neurons):
        acronym = meta_data.acronym[neuron_id]
        level = meta_data.level6[neuron_id]
        if not acronym == acronym_subtype:
            print(f"Skipping {neuron_id} because it is {acronym}")
            continue

        print(neuron_id, "/", num_neurons)

        fig_name = "pca_exp_data_%d_%s_%s" % (neuron_id, level, acronym.replace("/", ""))
        waveform = waveforms[neuron_id]
        temporal_waveform, spatial_waveform = single_waveform_pca(waveform, exp_tvec, fig_name, fig_folder)
        temporal_shapes.append(temporal_waveform)
        spatial_shapes.append(spatial_waveform)
    np.save(join(exp_data_folder, "exp_PCA_temporal_%s.npy" % acronym_subtype), temporal_shapes)
    np.save(join(exp_data_folder, "exp_PCA_spatial_%s.npy" % acronym_subtype), spatial_shapes)


def analyse_pca_simulated_data():
    sim_dt = 2**-5
    sim_name = "BBP"
    fig_folder = join(sim_data_folder, "pca_figures_%s" % sim_name)
    os.makedirs(fig_folder, exist_ok=True)
    sim_waveforms = np.load(join(sim_data_folder, "waveforms_sim_%s.npy" % sim_name))


    try:
        sim_waveforms_celltypes = np.load(join(sim_data_folder, "waveforms_sim_%s_celltype_list.npy" % sim_name))
    except FileNotFoundError:
        sim_waveforms_celltypes = None
    print(sim_waveforms.shape)
    sys.exit()

    num_neurons = sim_waveforms.shape[0]
    sim_num_tsteps = sim_waveforms.shape[1]
    sim_tvec = np.arange(sim_num_tsteps) * sim_dt
    temporal_shapes = []
    spatial_shapes = []

    for neuron_id in range(num_neurons):
        waveform = sim_waveforms[neuron_id]
        fig_name = "pca_sim_%s_%d" % (sim_name, neuron_id)
        temporal_waveform, spatial_waveform = single_waveform_pca(waveform, sim_tvec, fig_name, fig_folder)
        temporal_shapes.append(temporal_waveform)
        spatial_shapes.append(spatial_waveform)
    temporal_shapes = np.array(temporal_shapes)
    spatial_shapes = np.array(spatial_shapes)
    np.save(join(sim_data_folder, "hay_PCA_temporal.npy"), temporal_shapes)
    np.save(join(sim_data_folder, "hay_PCA_spatial.npy"), spatial_shapes)


def waveform_collection_PCA(temporal_shapes, tvec, fig_folder, fig_name):
    n_components = 3
    pca = PCA(n_components)
    projected = pca.fit_transform(temporal_shapes)

    temporal_shapes_new = pca.inverse_transform(projected)

    temporal_shapes_diff = temporal_shapes - temporal_shapes_new

    print(projected.shape)
    print(pca.components_.shape)
    # approx = np.zeros_like(waveform.T[:, :]) + pca.mean_[None, :]

    # for i in range(n_components):
    #    approx += np.dot(projected[:, i, None], pca.components_[None, i, :])
    plt.close("all")
    fig = plt.figure(figsize=[9, 9])
    ax1 = fig.add_subplot(531)
    ax2 = fig.add_subplot(532, aspect=1, xlim=[-1, 1.5], ylim=[-1, 1])
    ax2b = fig.add_subplot(533, aspect=1, xlim=[-1, 1.5], ylim=[-1, 1])

    ax3 = fig.add_subplot(512, ylim=[-0.7, 0.7])
    ax4 = fig.add_subplot(513, ylim=[-0.7, 0.7])
    ax5 = fig.add_subplot(514, ylim=[-0.7, 0.7])
    ax6 = fig.add_subplot(515)

    # ax3 = fig.add_subplot(133)

    ax1.plot(pca.components_[0, :])
    ax1.plot(pca.components_[1, :])
    ax1.plot(pca.components_[2, :])

    ax2.scatter(projected[:, 0], projected[:, 1], s=1, c='k')
    ax2b.scatter(projected[:, 0], projected[:, 2], s=1, c='k')

    ax2.axhline(0, ls='--', lw=0.5)
    ax2.axvline(0, ls='--', lw=0.5)
    ax2b.axhline(0, ls='--', lw=0.5)
    ax2b.axvline(0, ls='--', lw=0.5)

    l = [ax3.plot(tvec, eap_) for eap_ in temporal_shapes[:50]]
    l = [ax4.plot(tvec, eap_) for eap_ in temporal_shapes_new[:50]]
    l = [ax5.plot(tvec, eap_) for eap_ in temporal_shapes_diff[:50]]

    rel_error = np.max(np.abs(temporal_shapes_diff), axis=1) / (
                np.max(temporal_shapes, axis=1) - np.min(temporal_shapes, axis=1))
    ax6.hist(rel_error, bins=50)

    plt.savefig(join(fig_folder, "%s.png" % fig_name))


def sim_waveform_collection_pca():
    sim_dt = 2**-5
    sim_name = "hay"
    temporal_shapes = np.load(join(sim_data_folder, "hay_PCA_temporal.npy"))
    num_neurons = temporal_shapes.shape[0]
    sim_num_tsteps = temporal_shapes.shape[1]
    sim_tvec = np.arange(sim_num_tsteps) * sim_dt
    waveform_collection_PCA(temporal_shapes, sim_tvec, sim_data_folder, "hay_collection_PCA")


def exp_waveform_collection_pca():
    sim_dt = 2**-5
    sim_name = "hay"
    temporal_shapes = np.load(join(sim_data_folder, "hay_PCA_temporal.npy"))
    num_neurons = temporal_shapes.shape[0]
    sim_num_tsteps = temporal_shapes.shape[1]
    sim_tvec = np.arange(sim_num_tsteps) * sim_dt
    waveform_collection_PCA(temporal_shapes, sim_tvec, sim_data_folder, "hay_collection_PCA")


def plot_spike_features_waveform_collection():
    fig_folder = join("..", "PCA", "exp")
    os.makedirs(fig_folder, exist_ok=True)
    acronym_subtype = None# "VISp5"
    name = "exp"
    if acronym_subtype is not None:
        name += "_" + acronym_subtype
    feature_dicts = []

    for neuron_id in range(num_neurons):
        acronym = meta_data.acronym[neuron_id]
        level = meta_data.level6[neuron_id]
        if level != "Isocortex":
            # print(f"Skipping {neuron_id} because it is {level}")
            continue
        if (acronym_subtype is not None) and (acronym != acronym_subtype):
            print(f"Skipping {neuron_id} because it is {acronym}")
            continue
        feature_dict = extract_spike_features(waveforms[neuron_id], exp_tvec,
                               "eap_features_exp_%d" % (neuron_id),
                                              fig_folder, plot_it=False)
        # if feature_dict is not None:
        feature_dicts.append(feature_dict)

    # np.save(join(exp_data_folder, "EAP_feature_dicts_exp_%s.npy" % acronym_subtype), feature_dicts)
    np.save(join(exp_data_folder, "EAP_feature_dicts_%s.npy" % name), feature_dicts)


def plot_spike_features_waveform_collection_sim():
    sim_name = "BBP"
    sim_dt = 2**-5
    fig_folder = join("..", "PCA", sim_name)
    sim_waveforms = np.load(join(sim_data_folder, "waveforms_sim_%s.npy" % sim_name))
    soma_distance_from_plane = np.load(join(sim_data_folder,
                                            "waveforms_sim_%s_soma_distance.npy" % sim_name))

    sim_num_tsteps = sim_waveforms.shape[1]
    sim_num_neurons = sim_waveforms.shape[0]
    sim_tvec = np.arange(sim_num_tsteps) * sim_dt
    os.makedirs(fig_folder, exist_ok=True)
    feature_dicts = []
    #acronym_subtype = "VISp5"
    for neuron_id in range(sim_num_neurons):
        print(neuron_id)
        #acronym = meta_data.acronym[neuron_id]
        #level = meta_data.level6[neuron_id]
        #if not level == "Isocortex":
        #    print(f"Skipping {neuron_id} because it is {level}")
        #    continue

        # if neuron_id != 3257:
        #     continue
        #if not acronym == acronym_subtype:
        #    print(f"Skipping {neuron_id} because it is {acronym}")
        #    continue
        feature_dict = extract_spike_features(sim_waveforms[neuron_id], sim_tvec,
                               "eap_features_%s_%d" % (sim_name, neuron_id),
                               fig_folder, soma_dist_from_plane=soma_distance_from_plane[neuron_id],
                                              plot_it=False)
        # feature_dicts.append(feature_dict)
        # if feature_dict is not None:
        feature_dicts.append(feature_dict)

    np.save(join(sim_data_folder, "EAP_feature_dicts_%s.npy" % sim_name), feature_dicts)


def pca_waveform_collection_features():
    # name = "exp_VISp5"
    name = "allen"
    # name = "hay"
    # name = "hallermann"
    # name = "BBP"
    # name = "allen"


    include_pops = None # ["L5E", "L5I"]
    population_color_dict = {
        'L1E': 'k',
        'L1I': 'gray',
        'L2/3E': 'green',
        'L2/3I': 'lightgreen',
        'L4E': 'r',
        'L4I': 'pink',
        'L5I': 'sandybrown',
        'L5E': 'saddlebrown',
        'L6E': 'blue',
        'L6I': 'cyan',
                        }
    population_names = population_color_dict.keys()

    data_folder = exp_data_folder if "exp" in name else sim_data_folder
    fig_folder = join(data_folder, "..")
    feature_dicts_full = np.load(join(data_folder, "EAP_feature_dicts_%s.npy" % name),
                            allow_pickle=True)

    if "exp" in name:
        colors = 'k'
        feature_dicts = feature_dicts_full
    else:

        soma_distance_from_plane_full = np.load(join(data_folder,
                                           "waveforms_sim_%s_soma_distance.npy" % name))

        if name == "hay" or name == "hallermann":
            celltypes = ["L5E"] * len(feature_dicts_full)
        else:
            celltypes = np.load(join(data_folder,
                                 "waveforms_sim_%s_celltype_list.npy" % name))
        print(celltypes[0])
        feature_dicts = []
        colors = []
        markers = []
        sizes = []
        population_list = []
        for c_idx, celltype in enumerate(celltypes):
            if name == "allen":
                layer_ = "L" + celltype.split(", layer ")[-1]
                cell_class = "I" if "aspiny" in celltype else "E"
                pop_name = layer_.replace("a", "") + cell_class
                c = population_color_dict[pop_name]
            elif name == "BBP":
                layer_ = celltype.split("_")[0]
                layer_ = layer_.replace("L23", "L2/3")
                cell_class = "E" if ("PC" in celltype) or ("SS" in celltype) or ("SP" in celltype) else "I"
                pop_name = layer_ + cell_class
                c = population_color_dict[pop_name]
            elif name == "hay" or name == "hallermann":
                pop_name = celltype
                c = soma_distance_from_plane_full[c_idx]
            else:
                raise RuntimeError("cell_name not found!")
            if include_pops is not None:
                if pop_name in include_pops:
                    population_list.append(pop_name)
                    colors.append(c)
                    feature_dicts.append(feature_dicts_full[c_idx])
                    markers.append('o' if "aspiny" in celltype else "+")
                    sizes.append(5)
            else:
                population_list.append(pop_name)
                colors.append(c)
                feature_dicts.append(feature_dicts_full[c_idx])
                markers.append('o' if "aspiny" in celltype else "+")
                sizes.append(5)


    keys = feature_dicts[0].keys()
    features = np.zeros((len(feature_dicts), len(keys)))
    for idx in range(len(feature_dicts)):
        # print(feature_dicts[idx])
        features[idx, :] = [feature_dicts[idx][key] for key in keys]
        # print(features)

    n_components = 3
    pca = PCA(n_components)
    projected = pca.fit_transform(features)

    features_new = pca.inverse_transform(projected)

    # temporal_shapes_diff = temporal_shapes - temporal_shapes_new

    t_sne = manifold.TSNE(
        n_components=3,
        learning_rate="auto",
        perplexity=30,
        n_iter=250,
        init="random",
        #random_state=rng,
    )
    S_t_sne = t_sne.fit_transform(features)

    # print(S_t_sne)
    # print(projected.shape)
    # print(pca.components_.shape)
    # approx = np.zeros_like(waveform.T[:, :]) + pca.mean_[None, :]

    # for i in range(n_components):
    #    approx += np.dot(projected[:, i, None], pca.components_[None, i, :])
    plt.close("all")
    import mpl_toolkits.mplot3d
    fig = plt.figure(figsize=[16, 10])
    fig.subplots_adjust(left=0.07, right=0.97, top=0.96, bottom=0.15, hspace=0.4, wspace=0.4)
    ax1 = fig.add_subplot(321, title="PCA")
    ax2 = fig.add_subplot(323, xlabel="comp. 1", ylabel="comp. 2")
    ax3 = fig.add_subplot(325, xlabel="comp. 1", ylabel="comp. 3")

    ax2_tSNE = fig.add_subplot(222, title="t-SNE", xlabel="comp. 1", ylabel="comp. 2")
    ax3_tSNE = fig.add_subplot(224, title="t-SNE", xlabel="comp. 1", ylabel="comp. 3")
    ax2_tSNE.scatter(S_t_sne[:, 0], S_t_sne[:, 1], s=3, c=colors)
    ax3_tSNE.scatter(S_t_sne[:, 0], S_t_sne[:, 2], s=3, c=colors)

    # ax3 = fig.add_subplot(133)
    ax1.plot(pca.components_[0, :])
    ax1.plot(pca.components_[1, :])
    ax1.plot(pca.components_[2, :])

    ax2.scatter(projected[:, 0], projected[:, 1], s=3, c=colors)
    ax3.scatter(projected[:, 0], projected[:, 2], s=3, c=colors)

    # ax2.axhline(0, ls='--', lw=0.5)
    # ax2.axvline(0, ls='--', lw=0.5)
    # ax3.axhline(0, ls='--', lw=0.5)
    # ax3.axvline(0, ls='--', lw=0.5)

    # l = [ax3.plot(tvec, eap_) for eap_ in temporal_shapes[:50]]
    # l = [ax4.plot(tvec, eap_) for eap_ in temporal_shapes_new[:50]]
    # l = [ax5.plot(tvec, eap_) for eap_ in temporal_shapes_diff[:50]]

    # rel_error = np.max(np.abs(temporal_shapes_diff), axis=1) / (
    #             np.max(temporal_shapes, axis=1) - np.min(temporal_shapes, axis=1))
    # ax6.hist(rel_error, bins=50)

    lines = []
    line_names = []
    if not "exp" in name:
        for population_name in population_names:
            l = ax3.scatter([], [], color=population_color_dict[population_name])
            lines.append(l)
            line_names.append(population_name)

        fig.legend(lines, line_names, frameon=False, loc=(0.1, 0.01), ncol=len(population_names))
    fig_name = name
    if include_pops is not None:
        for pop in include_pops:
            fig_name += "_" + pop
    plt.savefig(join(fig_folder, "PCA_features_%s.png" % fig_name))


def pca_waveform_collection_combined():

    data_sets = ["BBP", "allen", "hay", "hallermann", "exp"]

    population_color_dict = {
        "exp": 'k',
        "allen": 'r',
        "BBP": 'b',
        "hay": 'cyan',
        "hallermann": 'purple'
        }

    population_names = population_color_dict.keys()
    fig_folder = join(exp_data_folder, "..")
    features = []
    colors = []
    for data_set_name in data_sets:
        data_folder = exp_data_folder if "exp" in data_set_name else sim_data_folder

        feature_dicts = np.load(join(data_folder, "EAP_feature_dicts_%s.npy" % data_set_name),
                                allow_pickle=True)
        keys = feature_dicts[0].keys()

        for idx in range(len(feature_dicts)):
            features.append([feature_dicts[idx][key] for key in keys])
            colors.append(population_color_dict[data_set_name])

    features = np.array(features)
    n_components = 3
    pca = PCA(n_components)
    projected = pca.fit_transform(features)

    t_sne = manifold.TSNE(
        n_components=3,
        learning_rate="auto",
        perplexity=60,
        n_iter=500,
        init="random",
        #random_state=rng,
    )
    S_t_sne = t_sne.fit_transform(features)


    plt.close("all")
    import mpl_toolkits.mplot3d
    fig = plt.figure(figsize=[16, 10])
    fig.subplots_adjust(left=0.07, right=0.97, top=0.96, bottom=0.15, hspace=0.4, wspace=0.4)
    ax1 = fig.add_subplot(321, title="PCA")
    ax2 = fig.add_subplot(323, xlabel="comp. 1", ylabel="comp. 2")
    ax3 = fig.add_subplot(325, xlabel="comp. 1", ylabel="comp. 3")

    ax2_tSNE = fig.add_subplot(222, title="t-SNE", xlabel="comp. 1", ylabel="comp. 2")
    ax3_tSNE = fig.add_subplot(224, title="t-SNE", xlabel="comp. 1", ylabel="comp. 3")
    ax2_tSNE.scatter(S_t_sne[:, 0], S_t_sne[:, 1], s=3, c=colors)
    ax3_tSNE.scatter(S_t_sne[:, 0], S_t_sne[:, 2], s=3, c=colors)

    # ax3 = fig.add_subplot(133)
    ax1.plot(pca.components_[0, :])
    ax1.plot(pca.components_[1, :])
    ax1.plot(pca.components_[2, :])

    ax2.scatter(projected[:, 0], projected[:, 1], s=3, c=colors)
    ax3.scatter(projected[:, 0], projected[:, 2], s=3, c=colors)


    lines = []
    line_names = []

    for population_name in data_sets:
        l = ax3.scatter([], [], color=population_color_dict[population_name])
        lines.append(l)
        line_names.append(population_name)

    fig.legend(lines, line_names, frameon=False, loc=(0.1, 0.01), ncol=len(population_names))

    plt.savefig(join(fig_folder, "PCA_features_combined.png"))


def plot_waveform_feature_separability():

    # name = "hay"
    # name = "hallermann"
    # name = "allen"
    name = "BBP"
    # name = "exp_VISp5"
    # name = "exp"
    include_pops = ["L5E", "L5I"]
    population_color_dict = {
        'L1E': 'k',
        'L1I': 'gray',
        'L2/3E': 'green',
        'L2/3I': 'lightgreen',
        'L4E': 'r',
        'L4I': 'pink',
        'L5I': 'sandybrown',
        'L5E': 'saddlebrown',
        'L6E': 'blue',
        'L6I': 'cyan',
                        }
    population_names = population_color_dict.keys()

    data_folder = exp_data_folder if "exp" in name else sim_data_folder
    fig_folder = join(data_folder, "..")
    feature_dicts_full = np.load(join(data_folder, "EAP_feature_dicts_%s.npy" % name),
                            allow_pickle=True)

    plot_lim_dict = {"p2p": [0, 1000],
                     "max_peak": [200, -1000],
                     "prepeak": [-20, 200],
                     "fwhm": [0, 0.6],
                     "postpeak_delay": [0, 2],
                     "decay": [0, 200]}

    if "exp" in name:
        colors = 'k'
        feature_dicts = feature_dicts_full
    else:
        soma_distance_from_plane_full = np.load(join(data_folder,
                                                "waveforms_sim_%s_soma_distance.npy" % name))
        if name == "hay" or name == "hallermann":
            celltypes = ["L5E"] * len(feature_dicts_full)
        else:
            celltypes = np.load(join(data_folder,
                                 "waveforms_sim_%s_celltype_list.npy" % name))
        print(celltypes[0])
        feature_dicts = []
        colors = []
        markers = []
        sizes = []
        population_list = []
        for c_idx, celltype in enumerate(celltypes):
            if name == "allen":
                layer_ = "L" + celltype.split(", layer ")[-1]
                cell_class = "I" if "aspiny" in celltype else "E"
                pop_name = layer_.replace("a", "") + cell_class
                c = population_color_dict[pop_name]
            elif name == "BBP":
                layer_ = celltype.split("_")[0]
                layer_ = layer_.replace("L23", "L2/3")
                cell_class = "E" if ("PC" in celltype) or ("SS" in celltype) or ("SP" in celltype) else "I"
                pop_name = layer_ + cell_class
                c = population_color_dict[pop_name]
            elif name == "hay" or name == "hallermann":
                pop_name = celltype
                c = soma_distance_from_plane_full[c_idx]
            else:
                raise RuntimeError("cell_name not found!")
            if include_pops is not None:
                if pop_name in include_pops:
                    population_list.append(pop_name)
                    colors.append(c)
                    feature_dicts.append(feature_dicts_full[c_idx])
                    markers.append('o' if "aspiny" in celltype else "+")
                    sizes.append(5)
            else:
                population_list.append(pop_name)
                colors.append(c)
                feature_dicts.append(feature_dicts_full[c_idx])
                markers.append('o' if "aspiny" in celltype else "+")
                sizes.append(5)

    keys = feature_dicts[0].keys()
    print(keys)
    features = np.zeros((len(feature_dicts), len(keys)))
    for idx in range(len(feature_dicts)):
        # print(feature_dicts[idx])
        features[idx, :] = [feature_dicts[idx][key] for key in keys]
        # print(features)

    fig = plt.figure(figsize=[16, 10])
    fig.subplots_adjust(left=0.07, right=0.97, top=0.96, bottom=0.15, hspace=0.8, wspace=0.8)
    for k1_idx, key1 in enumerate(keys):
        for k2_idx, key2 in enumerate(keys):
            if k1_idx == k2_idx:
                ax_ = fig.add_subplot(len(keys), len(keys), k1_idx + 1 + len(keys) * k2_idx, xlim=plot_lim_dict[key1])
                if "exp" in name:
                    pop_features_ = [feature[k1_idx] for f_idx, feature in enumerate(features)]
                    ax_.hist(pop_features_, color='k', bins=10, density=True)
                else:
                    for population_name in population_names:
                        pop_features_ = [feature[k1_idx] for f_idx, feature in enumerate(features)
                                         if population_list[f_idx] == population_name]
                        ax_.hist(pop_features_, color=population_color_dict[population_name], density=True)

            else:
                ax_ = fig.add_subplot(len(keys), len(keys), k1_idx + 1 + len(keys) * k2_idx, xlim=plot_lim_dict[key1],
                                      ylim=plot_lim_dict[key2])
                ax_.scatter(features[:, k1_idx], features[:, k2_idx], c=colors, s=3)
            if k1_idx == 0:
                ax_.set_ylabel(key2)
            if k2_idx == len(keys) - 1:
                ax_.set_xlabel(key1)
    lines = []
    line_names = []
    if not "exp" in name:
        for population_name in population_names:
            l = ax_.scatter([], [], color=population_color_dict[population_name])
            lines.append(l)
            line_names.append(population_name)

        fig.legend(lines, line_names, frameon=False, loc=(0.1, 0.01), ncol=len(population_names))

    simplify_axes(fig.axes)
    fig_name = name
    if include_pops is not None:
        for pop in include_pops:
            fig_name += "_" + pop
    plt.savefig(join(fig_folder, "feature_separability_%s.png" % fig_name), dpi=150)





if __name__ == '__main__':
    # plot_all_waveforms()
    # analyze_elec_size_SNR()
    # analyse_pca_exp_data()
    # analyse_pca_simulated_data()
    # sim_waveform_collection_pca()
    # plot_spike_features_waveform_collection()
    # plot_spike_features_waveform_collection_sim()
    # pca_waveform_collection_features()
    # pca_waveform_collection_combined()
    # plot_waveform_feature_separability()
    # analyse_waveform_collection(waveforms, exp_tvec, "exp_data")
    # analyse_simulated_waveform_collections()
    # animate_NPUltraWaveform(waveforms[54], exp_tvec, "anim_exp_54", join(fig_folder, "..", "anim"))
    animate_sim_waveform()

