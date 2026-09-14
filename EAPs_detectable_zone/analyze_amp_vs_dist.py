import numpy as np
import os
from os.path import join
import matplotlib.pyplot as plt
from plotting_convention import mark_subplots, simplify_axes


root_folder = os.path.abspath(join(os.path.dirname(__file__), '..'))

exp_data_folder = join(root_folder, "exp_data", "NPUltraWaveforms")
sim_data_folder = join(root_folder, "exp_data", "simulated")

data_set_name = "allen"

fig_folder = join(exp_data_folder, "figures")
os.makedirs(fig_folder, exist_ok=True)

x = np.load(join(exp_data_folder, "channels.xcoords.npy"))[:, 0]
z = np.load(join(exp_data_folder, "channels.ycoords.npy"))[:, 0]

ws = np.load(join(sim_data_folder, f"waveforms_sim_{data_set_name}_l23_elec_d_20um.npy"))
ct = np.load(join(sim_data_folder, f"waveforms_sim_{data_set_name}_celltype_list_l23_elec_d_20um.npy"))
soma_locs = np.load(join(sim_data_folder, f"waveforms_sim_{data_set_name}_soma_location_l23_elec_d_20um.npy"))

num_spikes = ws.shape[0]
num_tsteps = ws.shape[1]
num_elecs = ws.shape[2]

fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(131, xlabel="x (µm)", ylabel="z (µm)", aspect=1, xlim=[-25, 75], ylim=[-25, 300])

ax2_spiny = fig.add_subplot(232, xlabel="distance from soma (µm)", ylabel="Max amplitude (µV)", title="spiny cells",
                            xlim=(0, 100), ylim=(1, 200))
ax2_aspiny = fig.add_subplot(233, xlabel="distance from soma (µm)", title="aspiny cells",
                             ylabel="Max amplitude (µV)", xlim=(0, 100), ylim=(1, 200))

ax2_spiny_log = fig.add_subplot(235, xlabel="distance from soma (µm)", ylabel="Max amplitude (µV)",
                                title="spiny cells", xlim=(0, 100), ylim=(1, 200))
ax2_aspiny_log = fig.add_subplot(236, xlabel="distance from soma (µm)", ylabel="Max amplitude (µV)",
                                 title="aspiny cells", xlim=(0, 100), ylim=(1, 200))

for spike_idx in range(num_spikes):

    if not "layer 2/3" in ct[spike_idx]:
        continue
    print(ct[spike_idx])
    if "aspiny" in ct[spike_idx]:
        c = 'red'
        ax_ = ax2_aspiny
        ax_log = ax2_aspiny_log

    elif "spiny" in ct[spike_idx]:
        c = 'blue'
        ax_ = ax2_spiny
        ax_log = ax2_spiny_log
    else:
        c = 'orange'

    if "perisomatic" in ct[spike_idx]:
        marker = '.'
    else:
        marker = '.'

    ax1.plot(soma_locs[spike_idx, 0], soma_locs[spike_idx, 2], 'o', c=c, ms=1)
    w = ws[spike_idx]
    amp = np.max(np.abs(w), axis=0)
    dists = np.sqrt((x - soma_locs[spike_idx, 0])**2 +
                    (soma_locs[spike_idx, 1])**2 +
                    (z - soma_locs[spike_idx, 2])**2)

    amp_mask = amp > 1e-10

    ax_.plot(dists[amp_mask], amp[amp_mask], marker, c=c, ms=2, alpha=0.5)
    ax_log.semilogy(dists[amp_mask], amp[amp_mask], marker, c=c, ms=2, alpha=0.5)

    # ax2.plot(soma_loc[spike_idx, 0], soma_loc[spike_idx, 1], 'o', c='k', ms=1)

ax1.scatter(x, z, s=4, c='k')
# ax2.scatter(x, np.zeros(x.shape), s=4, c='r')

simplify_axes(fig.axes)
fig.savefig(join(sim_data_folder, "allen_soma_locations_l23_elec_d_20um.png"))