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

ws = np.load(join(sim_data_folder, f"waveforms_sim_{data_set_name}_l23_elec_r_7um_tiny_axon.npy"))
ct = np.load(join(sim_data_folder, f"waveforms_sim_{data_set_name}_celltype_list_l23_elec_r_7um_tiny_axon.npy"))
soma_locs = np.load(join(sim_data_folder, f"waveforms_sim_{data_set_name}_soma_location_l23_elec_r_7um_tiny_axon.npy"))

num_spikes = ws.shape[0]
num_tsteps = ws.shape[1]
num_elecs = ws.shape[2]

fig = plt.figure(figsize=(10, 8))
fig.subplots_adjust(wspace=0.5, right=0.96, top=0.96, hspace=0.85)
ax1 = fig.add_subplot(131, xlabel="x (µm)", ylabel="z (µm)", aspect=1, xlim=[-25, 75], ylim=[-25, 300])

ax2_Pyr = fig.add_subplot(432, xlabel="distance from soma (µm)", ylabel="Max amp (µV)", title="Pyr",
                          xlim=(0, 50), ylim=(1, 120))
ax2_Sst = fig.add_subplot(435, xlabel="distance from soma (µm)", title="SST",
                          ylabel="Max amp (µV)", xlim=(0, 50), ylim=(1, 120))
ax3_Pvalb = fig.add_subplot(438, xlabel="distance from soma (µm)", title="Pvalb",
                          ylabel="Max amp (µV)", xlim=(0, 50), ylim=(1, 120))
ax4_Htr3a = fig.add_subplot(4,3,11, xlabel="distance from soma (µm)", title="Htr3a",
                          ylabel="Max amp (µV)", xlim=(0, 50), ylim=(1, 120))
ax_comp = fig.add_subplot(2,3,3, xlabel="distance from soma (µm)", title="all L2/3 cells",
                          ylabel="Max amp (µV)", xlim=(0, 25), ylim=(0, 15))
# ax_diam = fig.add_subplot(2,3,3, xlabel="soma diams (µm)", title="all L2/3 cells",
#                           ylabel="count #")


# ax2_spiny_log = fig.add_subplot(235, xlabel="distance from soma (µm)", ylabel="Max amplitude (µV)",
#                                 title="spiny cells", xlim=(0, 100), ylim=(1, 200))
# ax2_aspiny_log = fig.add_subplot(236, xlabel="distance from soma (µm)", ylabel="Max amplitude (µV)",
#                                  title="aspiny cells", xlim=(0, 100), ylim=(1, 200))

pyr_lines = ["Rorb", "Ntsr1", "Nr5a1", "Scnn1a", "Slc17a6", "Cux2"]

cre_clrs = {"Pyr": "red",
            "Sst": 'green',
            "Htr3a": "purple",
            "Pvalb": "blue"}

ax_dict = {"Pyr": ax2_Pyr,
           "Sst": ax2_Sst,
           "Pvalb": ax3_Pvalb,
           "Htr3a": ax4_Htr3a}

soma_radii_dict = {"Pyr": [],
                  "Sst": [],
                  "Pvalb": [],
                  "Htr3a": []}

for spike_idx in range(num_spikes):

    cre_line = ct[spike_idx].split(", ")[-1]
    if cre_line in pyr_lines:
        cre_label = "Pyr"
        if "aspiny" in ct[spike_idx]:
            print(ct[spike_idx])
            print("strange!")
            continue
    else:
        cre_label = cre_line

    soma_radii_dict[cre_label].append(soma_locs[spike_idx, -1] / 2)
    ax_ = ax_dict[cre_label]

    if cre_label in cre_clrs:
        c = cre_clrs[cre_label]
    else:
        c = 'orange'
        print(f"Unknown cre label: {cre_label}")

    if not "layer 2/3" in ct[spike_idx]:
        continue


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

    l_, = ax_.plot(dists[amp_mask], amp[amp_mask], marker, c=c, ms=2, alpha=0.5)
    ax_comp.plot(dists[amp_mask], amp[amp_mask], marker, c=c, ms=5)

    # ax_log.semilogy(dists[amp_mask], amp[amp_mask], marker, c=c, ms=2, alpha=0.5)

    # ax2.plot(soma_loc[spike_idx, 0], soma_loc[spike_idx, 1], 'o', c='k', ms=1)

fig.text(0.71, 0.4, "Electrode radii: 7.24 µm", ha="left", va="top")
fig.text(0.71, 0.36, "Soma radii:", ha="left", va="top")
for c_, cre_label in enumerate(soma_radii_dict):
    # ax_diam.hist(soma_diam_dict[cre_label], density=True, alpha=0.5)
    #print(f"{cre_label}: {np.mean(soma_diam_dict[cre_label]):.2f} +- {np.std(soma_diam_dict[cre_label]):.2f}")
    lab = f"{cre_label}: {np.mean(soma_radii_dict[cre_label]):.2f} ± {np.std(soma_radii_dict[cre_label]):.2f} µm"
    fig.text(0.73, 0.3 - c_ / 30, lab, ha='left', va='bottom')

ax1.scatter(x, z, s=4, c='k')
# ax2.scatter(x, np.zeros(x.shape), s=4, c='r')

simplify_axes(fig.axes)
mark_subplots(ax1, "A", ypos=1.02, xpos=0.02)
mark_subplots(fig.axes, ypos=1.07, xpos=-0.02)

fig.savefig(join(sim_data_folder, "allen_soma_locations_l23_elec_r_7um_tiny_axon.png"))