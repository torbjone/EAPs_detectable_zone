
import os
import sys
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from plotting_convention import mark_subplots, simplify_axes

# ae_data_folder = join("..", "autoencoder", "waveform-results")
sim_data_folder = join("..", "exp_data", "simulated")
ae_data_folder = join("..", "autoencoder", "new_results")
sampling_rate = 30000  # Hz
exp_dt = 1 / sampling_rate * 1000
exp_num_tsteps = 82
exp_tvec = np.arange(exp_num_tsteps) * exp_dt

dx = 6
dz = 6

exp_data_folder = join("..", "exp_data", "NPUltraWaveforms")
x = np.load(join(exp_data_folder, "channels.xcoords.npy"))[:, 0]
z = np.load(join(exp_data_folder, "channels.ycoords.npy"))[:, 0]

num_elecs = len(x)
grid_shape = (int(num_elecs / 8), 8)

x_grid = x.reshape(grid_shape)
z_grid = z.reshape(grid_shape)

def plot_difference(orig_eap, approx_eap, tvec, fig_name, fig_folder):

    max_error = np.max(np.abs((orig_eap - approx_eap)), axis=0)
    max_error_idx = np.argmax(max_error)

    neg_peak = np.min(orig_eap, axis=0)
    eap_norm = np.max(np.abs(orig_eap))

    xlim = [-2, np.max(np.abs(x)) + 2]
    ylim = [-2, np.max(np.abs(z)) + 2]
    plt.close("all")
    fig = plt.figure(figsize=[10, 10])

    ax3 = fig.add_axes([0.02, 0.02, 0.2, 0.95], aspect=1, xlim=xlim, ylim=ylim,
                       xticks=[], yticks=[], frameon=False, title="Original")
    ax4 = fig.add_axes([0.22, 0.02, 0.2, 0.95], aspect=1, xlim=xlim, ylim=ylim,
                       xticks=[], yticks=[], frameon=False, title="Reconstructed")
    ax5 = fig.add_axes([0.42, 0.02, 0.2, 0.95], aspect=1, xlim=xlim, ylim=ylim,
                       xticks=[], yticks=[], frameon=False, title="Difference")
    #ax6 = fig.add_axes([0.81, 0.1, 0.17, 0.12], title="largest spike",
    #                   xticks=[], yticks=[], frameon=False)
    #ax7 = fig.add_axes([0.81, 0.24, 0.17, 0.12], title="largest error",
    #                   xticks=[], yticks=[], frameon=False)
    #ax8 = fig.add_axes([0.83, 0.82, 0.13, 0.12], title="PCA spatial\ndecay",
    #                   xlim=[-10, 150], ylim=[np.max(p2p) * 1e-2, np.max(p2p) * 1.5],
    #                   xticks=[0, 50, 100, 150], xlabel="distance (µm)")
    ax9 = fig.add_axes([0.65, 0.66, 0.32, 0.30], title="original",
                       xticks=[], yticks=[], frameon=False)
    ax10 = fig.add_axes([0.65, 0.33, 0.32, 0.3], title="recreated",
                       xticks=[], yticks=[], frameon=False)
    ax11 = fig.add_axes([0.65, 0.02, 0.32, 0.3], title="difference",
                       xticks=[], yticks=[], frameon=False)
    #ax1.set_title("spatial PCA", y=1.0)
    #ax2.set_title("temporal PCA", y=1.0)
    #vmax = np.max(np.abs(projected[:, 0]))

    #img = ax1.contourf(x_grid, z_grid, approx_eap[:, 0].reshape(grid_shape), levels=25,
    #                   cmap="bwr", vmin=-vmax, vmax=vmax)
    #cax_t = fig.add_axes([0.18, 0.04, 0.01, 0.84])
    #cbar = plt.colorbar(img, cax=cax_t, label="µV")

    x_norm = 0.8 * dx / tvec[-1]
    y_norm = 0.9 * dz / eap_norm
    eap_clr = lambda eap_min: plt.cm.viridis(0.0 + eap_min / np.min(neg_peak))
    for elec_idx in range(num_elecs):
        x_ = x[elec_idx] + tvec * x_norm
        y_ = z[elec_idx] + orig_eap[:, elec_idx] * y_norm
        y_rec = z[elec_idx] + approx_eap[:, elec_idx] * y_norm
        y_diff = z[elec_idx] + (orig_eap[:, elec_idx] - approx_eap[:, elec_idx]) * y_norm

        ax3.plot(x_, y_, lw=1, c='k', zorder=10, clip_on=False)
        ax4.plot(x_, y_rec, lw=1, c='k', zorder=10, clip_on=False)
        ax5.plot(x_, y_diff, lw=1, c='k', zorder=10, clip_on=False)

        ax9.plot(tvec, orig_eap[:, elec_idx], zorder=-neg_peak[elec_idx],
                  c=eap_clr(neg_peak[elec_idx]))
        ax10.plot(tvec, approx_eap[:, elec_idx], zorder=-neg_peak[elec_idx],
                 c=eap_clr(neg_peak[elec_idx]))
        ax11.plot(tvec, orig_eap[:, elec_idx] - approx_eap[:, elec_idx], zorder=-neg_peak[elec_idx],
                 c=eap_clr(neg_peak[elec_idx]))

    #ax9.plot(tvec, -temporal_waveform / np.min(temporal_waveform) * eap_norm, 'k', lw=0.5, zorder=10000)
    #ax10.plot(tvec, -temporal_waveform / np.min(temporal_waveform) * eap_norm, 'k', lw=0.5, zorder=10000)

    # ax1.plot(x[max_idx], z[max_idx], 'x', c='k')

    #if n_components > 1:
    #    for n_idx in range(1, n_components):
    #        ax2.plot(pca.components_[n_idx, :], c='gray', clip_on=False)
    #ax2.plot(pca.components_[0, :], c='k', clip_on=False)

    # l3, = ax7.plot(tvec, orig_eap[:, max_error_idx] - approx[:, max_error_idx], 'r')
    # l1, = ax7.plot(tvec, orig_eap[:, max_error_idx], 'k')
    # l2, = ax7.plot(tvec, approx_eap[:, max_error_idx], 'gray')
    # ax7.plot([tvec[-1] + 0.1, tvec[-1] + 0.1], [-eap_norm, 0], lw=1, c='k')
    # ax7.text(tvec[-1], -eap_norm / 2, "{:1.0f}\nµV".format(eap_norm), ha='right', va="center")

    # l3, = ax6.plot(tvec, orig_eap[:, max_idx_p2p] -approx_eap[:, max_idx_p2p], 'r')
    # l1, = ax6.plot(tvec, orig_eap[:, max_idx_p2p], 'k')
    # l2, = ax6.plot(tvec, approx_eap[:, max_idx_p2p], 'gray')
    # fig.legend([l1, l2, l3], ["original", "PCA rec.", "difference"], ncol=1, frameon=False,
    #            loc=(0.8, 0.01))
    # ax6.plot([tvec[-1] + 0.1, tvec[-1] + 0.1], [-eap_norm, 0], lw=1, c='k')
    # ax6.text(tvec[-1], -eap_norm / 2, "{:1.0f}\nµV".format(eap_norm), ha='right', va="center")
    # ax2 = fig.add_axes([0.5, 0.2, 0.2, 0.2])
    # ax2.plot(tvec, waveform[:, max_idx], 'k')
    # ax2.plot(tvec, approx.T[:, max_idx], 'red', ls='-')

    #ax8.semilogy(dist_from_max, p2p, 'k.')

    #simplify_axes(ax8)
    fig.savefig(join(fig_folder, fig_name))


def plot_latent_space(lat_var, waveforms, fig_name, fig_folder):
    lat_size = lat_var.shape[1]

    max_p2p = np.max(np.max(waveforms, axis=1) - np.min(waveforms, axis=1), axis=-1)
    print(max_p2p.shape)
    print(lat_size)

    num_events = 1000

    plt.close("all")
    fig = plt.figure(figsize=[10, 10])
    fig.subplots_adjust(top=0.99, bottom=0.05, right=0.98, left=0.05)
    for col in range(lat_size):
        for row in range(lat_size):
            if row > col:
                continue
            ax_ = fig.add_subplot(lat_size, lat_size,
                                      col + 1 + row * lat_size, frameon=False, xticks=[], yticks=[])
            if row == col:
                ax_.hist(lat_var[:num_events, row], bins=100, color='k')
            else:
                img = ax_.scatter(lat_var[:num_events, col], lat_var[:num_events, row],
                            s=2, c=max_p2p[:num_events])
    cax = fig.add_axes([0.25, 0.25, 0.01, 0.25])
    plt.colorbar(img, cax=cax, label="µV")
    plt.savefig(join(fig_folder, fig_name))

def plot_reconstructions():

    # sim_name = "hay"
    # waveforms = np.load(join(sim_data_folder, "waveforms_sim_%s.npy" % sim_name))[:, :82, :]
    waveforms = np.load(join(exp_data_folder, "clusters.waveforms.npy"))



    # ae_name = "vae-10"
    ae_name = "autoencoder-12"

    rec_filename = "%s_reconstructions_exp.npy" % ae_name
    z_filename = "%s_z_exp.npy" % ae_name

    ae_file = join(ae_data_folder, rec_filename)
    ae_z_file = join(ae_data_folder, z_filename)

    ae_data = np.load(ae_file)
    ae_z = np.load(ae_z_file)

    fig_folder = join(ae_data_folder, "latent_space_figures")
    fig_name = f"{ae_name}_latent_space_exp.png"
    os.makedirs(fig_folder, exist_ok=True)

    print(ae_data.shape)
    print(waveforms.shape)
    print(ae_z.shape)

    # plot_latent_space(ae_z, waveforms, fig_name, fig_folder)

    fig_folder = join(ae_data_folder, "rec_figures")
    os.makedirs(fig_folder, exist_ok=True)

    for waveform_idx in range(100):
        #waveform_idx = 54
        fig_name = f"{ae_name}_waveform_{waveform_idx}_exp.png"
        plot_difference(waveforms[waveform_idx], ae_data[waveform_idx].reshape(82, 384), exp_tvec,
                    fig_name, fig_folder)


def investigate_simulated_data():
    sim_name = "hay"
    waveforms = np.load(join(sim_data_folder, "waveforms_sim_%s.npy" % sim_name))[:, :, :]
    p2p_idx = np.argmax(np.max(waveforms, axis=1) - np.min(waveforms, axis=1), axis=-1)

    max_xs = x[p2p_idx]
    max_zs = z[p2p_idx]

    #print(np.max(waveforms[:, :, p2p_idx], axis=1) - np.min(waveforms[:, :, p2p_idx], axis=1))
    eap_features = np.load(join(sim_data_folder, "EAP_feature_dicts_%s.npy" % sim_name), allow_pickle=True)
    cell_depth = np.load(join(sim_data_folder, "waveforms_sim_%s_soma_distance.npy" % sim_name))
    num_spikes = len(eap_features)
    # print(eap_features)
    # ae_name = "vae-10"
    ae_name = "autoencoder-10"

    rec_filename = "%s_reconstructions_sim.npy" % ae_name
    z_filename = "%s_z_sim.npy" % ae_name

    ae_file = join(ae_data_folder, rec_filename)
    ae_z_file = join(ae_data_folder, z_filename)

    # ae_data = np.load(ae_file).reshape(376, 82, 384)
    # print(ae_data.shape)
    # p2p_idx_rec = np.argmax(np.max(ae_data, axis=1) - np.min(ae_data, axis=1), axis=-1)
    # max_xs_rec = x[p2p_idx_rec]
    # max_zs_rec = z[p2p_idx_rec]

    # plt.scatter(max_xs, max_xs_rec)
    # plt.show()

    ae_z = np.load(ae_z_file)

    features = list(eap_features[0].keys())

    feature_arrays = {key: [eap_features[i][key] for i in range(num_spikes)] for key in features}

    feature_arrays["soma_depth"] = cell_depth
    feature_arrays["x_pos"] = max_xs
    feature_arrays["z_pos"] = max_zs
    num_latent_variables = ae_z.shape[1]

    # print(ae_z.shape)
    num_cols = num_latent_variables
    num_rows = len(feature_arrays)

    # print(features)
    fig = plt.figure(figsize=[18, 10])
    fig.subplots_adjust(bottom=0.05, top=0.96, left=0.05, right=0.98, hspace=0.9, wspace=0.5)
    for row, key in enumerate(feature_arrays.keys()):
        for latent_idx in range(num_latent_variables):
            plot_idx = row * num_latent_variables + latent_idx + 1
            ax = fig.add_subplot(num_rows, num_cols, plot_idx, xlabel=key)

            ax.scatter(feature_arrays[key], ae_z[:, latent_idx], c='k', s=2)


    plt.savefig("hay_spikes_investigate.png", dpi=300)


if __name__ == '__main__':
    # investigate_simulated_data()
    plot_reconstructions()