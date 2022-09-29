import numpy as np
import os
from os.path import join
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from plotting_convention import mark_subplots, simplify_axes

exp_data_folder = join("..", "exp_data", "NPUltraWaveforms")
sim_data_folder = join("..", "exp_data", "simulated")

fig_folder = join(exp_data_folder, "figures")
os.makedirs(fig_folder, exist_ok=True)


x = np.load(join(exp_data_folder, "channels.xcoords.npy"))[:, 0]
z = np.load(join(exp_data_folder, "channels.ycoords.npy"))[:, 0]

spike_times = np.load(join(exp_data_folder, "spikes.times.npy"))
spike_clusters = np.load(join(exp_data_folder, "spikes.clusters.npy"))
waveforms = np.load(join(exp_data_folder, "clusters.waveforms.npy"))

depth_sort = np.argsort(z)

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
#print("Num elecs: ", num_elecs)

#print(x.shape)
#print(spike_times.shape)
#print(spike_clusters.shape)
#print(waveforms.shape)


def plot_NPUltraWaveform(waveform, tvec, fig_name, fig_folder, cell=None):

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
        from main import plot_superimposed_sec_type
        plot_superimposed_sec_type(cell, ax1)

    eap_norm = np.max(np.abs(waveform))

    max_peak_idxs = np.argmax(np.abs(waveform), axis=0)
    neg_peak = np.min(waveform, axis=0)

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
    fig.savefig(join(fig_folder, "waveform_%s.png" % fig_name), dpi=150)


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

    img = ax3.imshow(waveform[:, depth_sort].T, cmap="bwr",
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
                    cmap="bwr", vmin=-eap_norm, vmax=eap_norm)

    for t_idx in range(len(tvec)):
        print(t_idx + 1, len(tvec))
        lax3.set_xdata(tvec[t_idx])
        lax4.set_xdata(tvec[t_idx])
        for coll in img2.collections:
            coll.remove()

        img2 = ax2.contourf(x_grid, z_grid, waveform[t_idx, :].reshape(grid_shape), levels=25,
                                cmap="bwr", vmin=-eap_norm, vmax=eap_norm)
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
        return np.NaN
    if t1_idx is None:
        print("t1 is None!")
        t1_idx = len(eap)
        # plt.plot(eap)
        # plt.show()
        return np.NaN
    return (t1_idx - t0_idx) * dt


def analyse_waveform_collection(waveforms, tvec, name, celltypes_list=None):

    max_amps = []
    max_amp_xz = []
    max_amp_width = []
    num_elecs_above_threshold = []
    dt = tvec[1] - tvec[0]

    color_list = []

    for neuron_id in range(waveforms.shape[0]):
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
    plt.savefig("analysis_waveforms_%s.png" % name)
    plt.savefig("analysis_waveforms_%s.pdf" % name, dpi=100)


def analyse_simulated_waveform_collections():
    sim_dt = 2**-5
    sim_names = ["hallermann", "allen", "hay", "BBP"][1:2]
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
        analyse_waveform_collection(sim_waveforms, sim_tvec, "sim_%s_data" % sim_name, sim_waveforms_celltypes)


def plot_all_waveforms():

    for neuron_id in range(num_neurons):
        print(neuron_id, "/", num_neurons)
        fig_name = "exp_data_%d" % neuron_id
        waveform = waveforms[neuron_id]
        plot_NPUltraWaveform(waveform, exp_tvec, fig_name, fig_folder)
        # if neuron_id > 2:
        #      break


def animate_sim_waveform():
    sim_dt = 2**-5
    sim_name = ["hallermann", "allen", "hay", "BBP"][-1]
    waveform_idx = 5987
    sim_waveforms = np.load(join(sim_data_folder, "waveforms_sim_%s.npy" % sim_name))
    sim_num_tsteps = sim_waveforms.shape[1]
    sim_tvec = np.arange(sim_num_tsteps) * sim_dt
    fig_folder_sim = join(sim_data_folder, "anim", sim_name)
    animate_NPUltraWaveform(sim_waveforms[waveform_idx],
                            sim_tvec, "anim_%s_%d" % (sim_name, waveform_idx),
                            join(fig_folder_sim))


if __name__ == '__main__':
    # plot_all_waveforms()
    # analyse_waveform_collection(waveforms, exp_tvec, "exp_data")
    # analyse_simulated_waveform_collections()
    animate_NPUltraWaveform(waveforms[54], exp_tvec, "anim_exp_54", join(fig_folder, "..", "anim"))
    # animate_sim_waveform()

