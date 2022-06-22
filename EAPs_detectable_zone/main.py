import os
from os.path import join
import json
import numpy as np
import neuron
import LFPy
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
import elephant
import pandas as pd
from plotting_convention import mark_subplots, simplify_axes


cell_models_folder = join("cell_models")
sigma = 0.3  # S/m

cell_rot_dict = {515175253: [-np.pi/2 * 0.9, np.pi/10, 0],
                 488462783: [-np.pi/3, -np.pi/4, np.pi * 0.8]}
np.random.seed(1234)


def save_neural_spike_data(cell, data_folder, sim_name, elec_params):

    grid_electrode = LFPy.RecExtElectrode(cell, **elec_params)
    eaps = grid_electrode.get_transformation_matrix() @ cell.imem * 1e3
    os.makedirs(data_folder, exist_ok=True)
    np.save(join(data_folder, 'tvec_{}.npy'.format(sim_name)), cell.tvec)
    np.save(join(data_folder, 'vmem_{}.npy'.format(sim_name)), cell.vmem)
    np.save(join(data_folder, 'imem_{}.npy'.format(sim_name)), cell.imem)
    np.save(join(data_folder, 'eaps_{}.npy'.format(sim_name)), eaps)

    np.save(join(data_folder, 'x_{}.npy'.format(sim_name)), cell.x)
    np.save(join(data_folder, 'y_{}.npy'.format(sim_name)), cell.y)
    np.save(join(data_folder, 'z_{}.npy'.format(sim_name)), cell.z)
    np.save(join(data_folder, 'd_{}.npy'.format(sim_name)), cell.d)


def load_neural_data(data_folder, sim_name):

    cell_dict = {}
    cell_dict["tvec"] = np.load(join(data_folder, 'tvec_{}.npy'.format(sim_name)))
    cell_dict["vmem"] = np.load(join(data_folder, 'vmem_{}.npy'.format(sim_name)))
    cell_dict["imem"] = np.load(join(data_folder, 'imem_{}.npy'.format(sim_name)))
    cell_dict["eaps"] = np.load(join(data_folder, 'eaps_{}.npy'.format(sim_name)))

    cell_dict["x"] = np.load(join(data_folder, 'x_{}.npy'.format(sim_name)))
    cell_dict["y"] = np.load(join(data_folder, 'y_{}.npy'.format(sim_name)))
    cell_dict["z"] = np.load(join(data_folder, 'z_{}.npy'.format(sim_name)))
    cell_dict["d"] = np.load(join(data_folder, 'd_{}.npy'.format(sim_name)))

    return cell_dict


def return_allen_cell_model(model_folder, dt, tstop):
    mod_folder = join(model_folder, "modfiles")
    if not os.path.isdir(join(mod_folder, "x86_64")):
        print("Compiling mechanisms ...")
        cwd = os.getcwd()
        os.chdir(mod_folder)
        os.system("nrnivmodl")
        os.chdir(cwd)

    neuron.load_mechanisms(mod_folder)

    model_file = join(model_folder, "fit_parameters.json")
    manifest_file = join(model_folder, "manifest.json")
    metadata_file = join(model_folder, "model_metadata.json")
    morph_file = join(model_folder, "reconstruction.swc")

    params = json.load(open(model_file, 'r'))
    manifest = json.load(open(manifest_file, 'r'))
    metadata = json.load(open(metadata_file, 'r'))
    model_type = manifest["biophys"][0]["model_type"]

    # print(model_type)
    Ra = params["passive"][0]["ra"]

    if model_type == "Biophysical - perisomatic":
        e_pas = params["passive"][0]["e_pas"]
        cms = params["passive"][0]["cm"]

    celsius = params["conditions"][0]["celsius"]
    reversal_potentials = params["conditions"][0]["erev"]
    v_init = params["conditions"][0]["v_init"]
    active_mechs = params["genome"]
    neuron.h.celsius = celsius
    # print(Ra, celsius, v_init)
    # print(reversal_potentials)
    # print(active_mechs)

    # Define cell parameters
    cell_parameters = {
        'morphology': morph_file,
        'v_init': -70,    # initial membrane potential
        'passive': False,   # turn on NEURONs passive mechanism for all sections
        'nsegs_method': 'fixed_length', # spatial discretization method
        'max_nsegs_length': 20.,
        #'lambda_f' : 200.,           # frequency where length constants are computed
        'dt': dt,      # simulation time step size
        'tstart': -100,      # start time of simulation, recorders start at t=0
        'tstop': tstop,
        'pt3d': True,
        'custom_code': [join(cell_models_folder, 'remove_axon.hoc')]
    }

    cell = LFPy.Cell(**cell_parameters)
    cell.metadata = metadata
    cell.manifest = manifest
    # cell.set_rotation(z=np.pi/1.25)
    #neuron.h.load_file('stdrun.hoc')
    #cvode = neuron.h.CVode()
    #cvode.active(1)
    # cvode = neuron.h.CVode()
    #neuron.h.cvode.cache_efficient(1)

    for sec in neuron.h.allsec():
        sectype = sec.name().split("[")[0]
        sec.insert("pas")
        if model_type == "Biophysical - perisomatic":
            sec.e_pas = e_pas
            for cm_dict in cms:
                if cm_dict["section"] == sectype:
                    exec("sec.cm = {}".format(cm_dict["cm"]))
        sec.Ra = Ra

        for sec_dict in active_mechs:
            if sec_dict["section"] == sectype:
                # print(sectype, sec_dict)
                if not sec_dict["mechanism"] == "":

                    if not sec.has_membrane(sec_dict["mechanism"]):
                        sec.insert(sec_dict["mechanism"])
                        # print("Inserted ", sec_dict["mechanism"])
                exec("sec.{} = {}".format(sec_dict["name"], sec_dict["value"]))

        for sec_dict in reversal_potentials:
            if sec_dict["section"] == sectype:
                # print(sectype, sec_dict)
                for key in sec_dict.keys():
                    if not key == "section":
                        exec("sec.{} = {}".format(key, sec_dict[key]))

    # for sec in neuron.h.allsec():
    #     if hasattr(sec, "eca"):
    #         print(sec.cao)
    #         sec.cao = 2
            # print(sec.name(), sec.eca)
    #print(cell.metadata["id"], cell.metadata["id"] in cell_rot_dict)
    if cell.metadata["id"] in cell_rot_dict:
        print("Manual rot")
        cell.set_rotation(x=cell_rot_dict[cell.metadata["id"]][0],
                          y=cell_rot_dict[cell.metadata["id"]][1],
                          z=cell_rot_dict[cell.metadata["id"]][2])
    else:
        from rotation_lastis import find_major_axes, alignCellToAxes
        axes = find_major_axes(cell)
        alignCellToAxes(cell, axes[2], axes[1])

    # cell.set_rotation(x=-np.pi/2, y=-np.pi/5)
    #cell.set_pos(z=-np.max(cell.z) - 5)
    neuron.h.secondorder = 0
    return cell#, model_type


def insert_current_stimuli(cell, amp):
    stim_params = {'amp': amp,
                   'idx': 0,
                   'pptype': "ISyn",
                   'dur': 1e9,
                   'delay': 0}
    neuron.load_mechanisms(cell_models_folder)
    synapse = LFPy.StimIntElectrode(cell, **stim_params)
    return synapse, cell


def return_spike_time_idxs(vm):
    """Returns threshold crossings for membrane
    potential of single compartment"""
    # num_tsteps_in_half_ms = int(0.5 / self.dt)
    crossings = []
    threshold = -20

    if np.max(vm) < threshold:
        return np.array([])
    for t_idx in range(1, len(vm)):
        if vm[t_idx - 1] < threshold <= vm[t_idx]:
            crossings.append(t_idx)

    return np.array(crossings)


def extract_spike(cell):
    spike_window = [-1, 4]
    spike_time_idxs = return_spike_time_idxs(cell.somav)
    # Use last spike, if it is not cut off
    if cell.tvec[spike_time_idxs[-1]] + spike_window[1] <= cell.tvec[-1]:
        t0 = cell.tvec[spike_time_idxs[-1]] + spike_window[0]
        t1 = cell.tvec[spike_time_idxs[-1]] + spike_window[1]
        used_idx = spike_time_idxs[-1]
    elif len(spike_time_idxs) > 2:
        t0 = cell.tvec[spike_time_idxs[-2]] + spike_window[0]
        t1 = cell.tvec[spike_time_idxs[-2]] + spike_window[1]
        used_idx = spike_time_idxs[-2]
    else:
        # t0 = cell.tvec[spike_time_idxs[-2]] + spike_window[0]
        # t1 = cell.tvec[spike_time_idxs[-2]] + spike_window[1]
        t0 = cell.tvec[-1] - spike_window[1] + spike_window[0]
        t1 = cell.tvec[-1]
        used_idx = spike_time_idxs[-1]

    t0_idx = np.argmin(np.abs(cell.tvec - t0))
    t1_idx = np.argmin(np.abs(cell.tvec - t1))

    cell.tvec = cell.tvec[t0_idx:t1_idx] - cell.tvec[t0_idx]
    cell.vmem = cell.vmem[:, t0_idx:t1_idx]
    cell.imem = cell.imem[:, t0_idx:t1_idx]
    return used_idx


def find_good_stim_amplitude_allen(cell_name, model_folder, dt, tstop):
    amp = -0.2#-0.05
    num_spikes = 0
    min_spikes = 2
    max_spikes = 10

    while not min_spikes <= num_spikes <= max_spikes:
        print("Testing amp {:1.3f} on cell {}".format(amp, cell_name))
        if num_spikes < min_spikes:
            amp *= 1.5
        elif num_spikes > max_spikes:
            amp *= 0.75
        cell = return_allen_cell_model(model_folder, dt, tstop)
        synapse, cell = insert_current_stimuli(cell, amp)
        cell.simulate(rec_vmem=True, rec_imem=True)

        num_spikes = len(return_spike_time_idxs(cell.somav))
        if not min_spikes <= num_spikes <= max_spikes:
            synapse = None
            cell.__del__()

    return cell


def spatiotemporal_shape(ax, cell):

    # x0, x1, dx = -15, 15, 6
    #x0, x1, dx = 0, 3, 6
    z0, z1, dz = -33, 76, 6

    # x_grid, z_grid = np.meshgrid(np.arange(x0, x1 + dx, dx),
    #                              np.arange(z0, z1 + dz, dz))
    # y_grid = np.zeros(x_grid.shape)

    z_grid = np.arange(z0, z1 + dz, dz)
    y_grid = np.zeros(z_grid.shape)
    x_grid = np.zeros(z_grid.shape)

    grid_elec_params = {
        'sigma': sigma,  # Saline bath conductivity
        'x': x_grid.flatten(),  # electrode requires 1d vector of positions
        'y': y_grid.flatten(),
        'z': z_grid.flatten(),
        "method": "root_as_point",
    }
    # print(grid_elec_params["x"])
    # print(grid_elec_params["z"])
    #idxs_sorted = np.argsort(grid_elec_params["z"])

    grid_electrode = LFPy.RecExtElectrode(cell, **grid_elec_params)
    eaps = grid_electrode.get_transformation_matrix() @ cell.imem * 1e3

    vmax = np.max(np.abs(eaps))

    img = ax.imshow(eaps, vmin=-vmax, vmax=vmax,
              cmap="seismic_r", origin="lower", extent=[0., cell.tvec[-1],
                                                        grid_elec_params["z"][0],
                                                        grid_elec_params["z"][-1]])

    cax = ax.get_figure().add_axes([ax.axes.get_position().x1 + 0.05,
                                    0.7,
                                    0.007,
                                    0.25])

    cbar = plt.colorbar(img, cax=cax, label="µV")

    ax.plot([3, 4], [-30, -30], c='k', lw=3)
    ax.text(3.5, -31, "1 ms", ha="center", va="top")

    ax.plot([4, 4], [-30, -20], c='k', lw=3)
    ax.text(4.1, -25, "10 µm", va="center", ha="left")


def detectable_volume(ax, cell, x_grid, y_grid, z_grid,
                      projection=('x', 'z')):
    zips = []
    for x1, x2 in cell.get_idx_polygons(projection=projection):
        zips.append(list(zip(x1, x2)))
    polycol = PolyCollection(zips,
                             edgecolors='none',
                             facecolors='gray',
                             rasterized=True)

    ax.add_collection(polycol)

    cax = ax.get_figure().add_axes([ax.axes.get_position().x1 + 0.01,
                                    ax.axes.get_position().y0,
                                    0.007,
                                    ax.axes.get_position().y1
                                    -ax.axes.get_position().y0])

    grid_elec_params = {
        'sigma': sigma,  # Saline bath conductivity
        'x': x_grid.flatten(),  # electrode requires 1d vector of positions
        'y': y_grid.flatten(),
        'z': z_grid.flatten(),
        "method": "root_as_point",
    }

    grid_electrode = LFPy.RecExtElectrode(cell, **grid_elec_params)
    eaps = grid_electrode.get_transformation_matrix() @ cell.imem * 1e3
    eaps_p2p = np.max(eaps, axis=1) - np.min(eaps, axis=1)

    #ax.plot([-100, -80], [-100, -100], lw=2, c='k')
    #ax.text(-90, -110, "20 µm", ha="center", va="top")
    detection_threshold = 30.

    levels_norm = [0, detection_threshold, 1e9]  # scale_max * levels
    colors_from_map = ['0.95', '#ffbbbb', (0.5, 0.5, 0.5, 1)]

    if projection[0] == 'x' and projection[1] == 'z':
        x1_grid = x_grid
        x2_grid = z_grid
    elif projection[0] == 'y' and projection[1] == 'z':
        x1_grid = y_grid
        x2_grid = z_grid
    elif projection[0] == 'x' and projection[1] == 'y':
        x1_grid = x_grid
        x2_grid = y_grid
    else:
        raise RuntimeError("Unaccepted projection!")

    ep_intervals = ax.contourf(x1_grid, x2_grid,
                                 eaps_p2p.reshape(x1_grid.shape),
                                 zorder=-2, colors=colors_from_map,
                                 levels=levels_norm, extend='both')
    cbar = plt.colorbar(ep_intervals, cax=cax)

    cbar.set_ticks(np.array([detection_threshold / 2, 1e9 / 2]))
    cbar.set_ticklabels(np.array(["<%d µV" % detection_threshold,
                                  ">%d µV" % detection_threshold]))


def run_chosen_allen_models():

    model_ids = [f.split('_')[-1] for f in os.listdir(cell_models_folder)
                  if f.startswith("neuronal_model_") and
                  os.path.isdir(join(cell_models_folder, f))][::-1]

    print(model_ids)

    # model_ids = [
    #     488462783,
    #     485720587,
    #     478047816,
    #     482934212,
    #     483108201,
    #     485513184,
    #     486508647,
    #     486509958,
    #     486558444,
    #     486909496,
    #     488083972,
    #     491766131,
    #     497229075,
    #     497229124,
    #     497232312,
    #     497232482,
    #     497232507,
    #     497232564,
    #     497232571,
    #     497232692,
    #     497232839,
    #     497232858,
    #     497232999,
    #     497233049,
    #     497233139,
    #     497233278,
    #     497233307,
    #     515175260,
    #     515175291,
    #     515175354,
    #     329321704,
    #     471087975,
    #     472299294,
    #     472300877,
    #     472451419,
    #     473834758,
    #     473862496,
    #     473863035,
    #     473863578,
    #     477880244,
    #     478809991,
    #     478513398,
    #     480630344,
    #     480633088,
    #              ]

    #print(model_ids)
    #model_ids = [f for f in model_ids if f.startswith("4972")]
    dt = 2**-7
    tstop = 120
    data_folder = join("..", "model_scan", "sim_data")
    fig_folder = join("..", "model_scan")

    for model_id in model_ids:
        #if not model_id == '483108201':
        # if not model_id == '486508647':
        #  continue
        #print("Running ", model_id)

        pid = os.fork()
        if pid == 0:

            model_folder = join(cell_models_folder, "neuronal_model_%s" % model_id)
            cell = return_allen_cell_model(model_folder, dt, tstop)
            model_type = cell.manifest["biophys"][0]["model_type"]
            cell_type = cell.metadata["specimen"]["specimen_tags"][1]["name"]
            cell_region = cell.metadata["specimen"]["structure"]["name"]

            #if "- spiny" in cell_type and "layer 1" not in cell_region:
            print("Running: ", model_id, model_type, cell_type, cell_region)
            # model_folder = join(cell_models_folder,  "neuronal_model_{}".format(model_id))
            cell.__del__()
            cell = find_good_stim_amplitude_allen(model_id, model_folder, dt, tstop)

            x0, x1, dx_hd = -50, 51, 2
            z0, z1, dz_hd = -50, 51, 2

            x0_ld, x1_ld, dx_ld = -30, 30, 20
            z0_ld, z1_ld, dz_ld = -30, 30, 20

            x_grid_ld, z_grid_ld = np.meshgrid(np.arange(x0_ld, x1_ld + dx_ld, dx_ld),
                                               np.arange(z0_ld, z1_ld + dz_ld, dz_ld))

            grid_elec_params_ld = {
                'sigma': sigma,  # Saline bath conductivity
                'x': x_grid_ld.flatten(),  # electrode requires 1d vector of positions
                'y': np.zeros(x_grid_ld.shape).flatten(),
                'z': z_grid_ld.flatten(),
                "method": "root_as_point",
            }

            #save_neural_spike_data(cell, data_folder, model_id, eaps)

            plt.close("all")
            fig = plt.figure(figsize=[16, 9])
            fig.suptitle("%s; %s; %s; %s" % (model_id, model_type,
                                             cell_type, cell_region),
                         size=14)
            ax1 = fig.add_axes([0.05, 0.32, 0.17, 0.67], aspect=1,
                               frameon=False, title="xz-plane",
                              xticks=[], yticks=[])

            ax2 = fig.add_axes([0.05, 0.17, 0.13, 0.13], frameon=False,
                               xticks=[], yticks=[])
            ax3 = fig.add_axes([0.05, 0.01, 0.13, 0.13], frameon=False,
                               xticks=[], yticks=[])

            ax4a = fig.add_axes([0.23, 0.65, 0.15, 0.25], frameon=False,
                               xticks=[], yticks=[], aspect=1, xlim=[x0, x1],
                               ylim=[z0, z1], title="xz-plane")
            ax4b = fig.add_axes([0.23, 0.35, 0.15, 0.25], frameon=False,
                               xticks=[], yticks=[], aspect=1, xlim=[x0, x1],
                               ylim=[z0, z1], title="yz-plane")
            ax4c = fig.add_axes([0.23, 0.05, 0.15, 0.25], frameon=False,
                               xticks=[], yticks=[], aspect=1, xlim=[x0, x1],
                               ylim=[z0, z1], title="xy-plane")

            ax5 = fig.add_axes([0.45, 0.05, 0.15, 0.9], frameon=False,
                               xticks=[], yticks=[],)

            ax6 = fig.add_axes([0.62, 0.01, 0.37, 0.9], frameon=False,
                               xticks=[], yticks=[],
                               xlim=[x0_ld - dz_ld / 4, x1_ld + dz_ld],
                               ylim=[z0_ld - dz_ld, z1_ld + dz_ld],
                               title="xz-plane",
                               aspect=1,)

            idx_apic = np.argmax(cell.z.mean(axis=1))
            idx_mid = cell.get_closest_idx(z=(np.max(cell.z) + np.min(cell.z))/2)
            idx_bottom = cell.get_closest_idx(z=np.min(cell.z))
            idx_soma = 0

            ax1.plot([-150, -50], [200, 200], c='k', lw=2)
            ax1.text(-100, 220, "100 µm", va="bottom", ha="center")

            for ax in [ax4a, ax4b, ax4c]:
                ax.plot([-30, -20], [35, 35], c='k', lw=2)
                ax.text(-25, 37, "10 µm", va="bottom", ha="center")

            ax2.text(1, -40, r"$V_{\rm m}$")
            ax2.set_yticks([cell.vmem[idx_soma, 0]])
            ax2.set_yticklabels(["{:1.0f} mV".format(cell.vmem[idx_soma, 0])])
            ax2.plot([cell.tvec[-1], cell.tvec[-1]], [0, -50], c='k', lw=3)
            ax2.text(cell.tvec[-1] + 2, -25, "50 mV")
            ax2.plot([50, 60], [-93, -93], c='k', lw=3)
            ax2.text(55, -95, "10 ms", va="top", ha="center")

            elec_params = {
                'sigma': sigma,  # Saline bath conductivity
                'x': np.array([10]),  # electrode requires 1d vector of positions
                'y': np.array([0]),
                'z': np.array([0]),
                'r': 5.,
                'N': [1, 0, 0],
                "method": "root_as_point",
            }
            electrode = LFPy.RecExtElectrode(cell, **elec_params)
            eaps_hd = electrode.get_transformation_matrix() @ cell.imem * 1e3

            ax3.text(1, np.min(eaps_hd[0])/2, r"$V_{\rm e}$")
            ax3.set_yticks([0])
            ax3.set_yticklabels(["0 µV"])
            ax3.plot([cell.tvec[-1], cell.tvec[-1]], [0, np.min(eaps_hd[0])], c='k', lw=3)
            ax3.text(cell.tvec[-1] + 2,  np.min(eaps_hd[0]) / 2, "{:1.0f} µV".format(np.abs(np.min(eaps_hd[0]))))

            ax1.plot(elec_params["x"], elec_params["z"], 'D', c='r', ms=9)
            #ax4a.plot(elec_params["x"], elec_params["z"], 'D', c='r', ms=9)

            ax1.plot(cell.x.T, cell.z.T, c='k', zorder=-5)
            ax1.plot(cell.x[idx_soma].mean(), cell.z[idx_soma].mean(), 'o', c='blue', ms=9)
            ax1.plot(cell.x[idx_mid].mean(), cell.z[idx_mid].mean(), 'o', c='green', ms=9)
            ax1.plot(cell.x[idx_bottom].mean(), cell.z[idx_bottom].mean(), 'o', c='orange', ms=9)
            ax1.plot(cell.x[idx_apic].mean(), cell.z[idx_apic].mean(), 'o', c='purple', ms=9)
            ax2.plot(cell.tvec, cell.vmem[idx_soma], c='blue')
            ax2.plot(cell.tvec, cell.vmem[idx_mid], "green")
            ax2.plot(cell.tvec, cell.vmem[idx_bottom], "orange")
            ax2.plot(cell.tvec, cell.vmem[idx_apic], "purple")
            ax3.plot(cell.tvec, eaps_hd[0], c='r')

            spike_time_idx = extract_spike(cell)
            ax3.axvline(spike_time_idx * dt, c='gray', lw=0.5, ls='--')
            ax2.axvline(spike_time_idx * dt, c='gray', lw=0.5, ls='--')

            spatiotemporal_shape(ax5, cell)
            ax5.set_yticks([0])
            ax5.set_yticklabels(["$z=0$"])
            ax5.axis("auto")

            x0, x1, dx = -50, 50, 2
            z0, z1, dz = -50, 50, 2

            x_grid, z_grid = np.meshgrid(np.arange(x0, x1 + dx, dx),
                                         np.arange(z0, z1 + dz, dz))
            y_grid = np.zeros(x_grid.shape)
            detectable_volume(ax4a, cell, x_grid, y_grid, z_grid,
                              projection=('x', 'z'))

            y_grid, z_grid = np.meshgrid(np.arange(x0, x1 + dx, dx),
                                         np.arange(z0, z1 + dz, dz))
            x_grid = np.zeros(y_grid.shape)
            detectable_volume(ax4b, cell, x_grid, y_grid, z_grid,
                              projection=('y', 'z'))

            x_grid, y_grid = np.meshgrid(np.arange(x0, x1 + dx, dx),
                                         np.arange(x0, x1 + dx, dx))
            z_grid = np.zeros(x_grid.shape)
            detectable_volume(ax4c, cell, x_grid, y_grid, z_grid,
                              projection=('x', 'y'))

            grid_electrode_ld = LFPy.RecExtElectrode(cell, **grid_elec_params_ld)
            eaps_ld = grid_electrode_ld.get_transformation_matrix() @ cell.imem * 1e3

            zips = []
            for x1, x2 in cell.get_idx_polygons(projection=('x', 'z')):
                zips.append(list(zip(x1, x2)))
            polycol = PolyCollection(zips,
                                     edgecolors='none',
                                     facecolors='0.9',
                                     rasterized=True)

            ax6.add_collection(polycol)
            for idx in range(len(eaps_ld)):
                x_ = grid_elec_params_ld["x"][idx]
                z_ = grid_elec_params_ld["z"][idx]
                t_ = x_ + cell.tvec / cell.tvec[-1] * dx_ld * 0.8
                eap_ = eaps_ld[idx] - eaps_ld[idx, 0]
                norm = dz_ld * 0.45 / np.max(np.abs(eap_))
                v_ = z_ + eap_ * norm

                ax6.plot([x_ + dx_ld * 0.83, x_ + dx_ld * 0.83],
                         [z_ + np.max(eap_) * norm, z_ + np.min(eap_) * norm], c='r', lw=2)
                ax6.text(x_ + dx_ld * 0.8, z_ + (np.max(eap_) + np.min(eap_)) * norm / 2,
                         "{:1.0f}\nµV".format(np.max(eap_) - np.min(eap_)),
                         ha="right", color='r', va="center")
                ax6.plot(x_, z_, 'o', c='k')
                ax6.plot(t_, v_, 'k')

                if idx == 0:
                    ax6.plot(np.array([x_ + 1 / cell.tvec[-1] * dx_ld * 0.8,
                                       x_ + 2 / cell.tvec[-1] * dx_ld * 0.8]),
                             [z_ + np.min(eap_) * norm, z_ + np.min(eap_) * norm], lw=2, c='k')
                    ax6.text(x_ + 1.5 / cell.tvec[-1] * dx_ld * 0.8,
                             z_ + np.min(eap_) * norm * 1.1, "1 ms", va="top", ha="center")

            ax6.plot([15, 25], [-45, -45], c='k', lw=3)
            ax6.text(20, -46, "10 µm", ha="center", va="top")

            mark_subplots(ax1, "A", ypos=0.97, xpos=0.0)
            mark_subplots(ax2, "B", ypos=0.97, xpos=0.0)
            mark_subplots(ax3, "C", ypos=0.97, xpos=0.0)
            mark_subplots(ax4a, "D", ypos=1.02, xpos=-0.05)
            mark_subplots(ax5, "E", ypos=0.99, xpos=-0.05)
            mark_subplots(ax6, "F", ypos=1.02, xpos=0.15)

            os.makedirs(fig_folder, exist_ok=True)
            plt.savefig(join(fig_folder, "spike_%s.png" % model_id), dpi=150)

            #else:
            #    print("Skipping: ",  model_id, model_type, cell_type, cell_region)
            # run_allen_example(model_id, run_sim=True)
            os._exit(0)
        else:
            os.waitpid(pid, 0)


if __name__ == '__main__':

    run_chosen_allen_models()

