import os
import sys
from os.path import join
from glob import glob
import posixpath
import json
import numpy as np
import neuron
import scipy.signal as ss
import LFPy
import matplotlib
# matplotlib.use("AGG")
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

imem_eap_folder = join("..", "imem_EAPs")
os.makedirs(imem_eap_folder, exist_ok=True)
hay_folder = join(cell_models_folder, "L5bPCmodelsEH")
bbp_folder = join(cell_models_folder, "bbp_models")
allen_folder = join(cell_models_folder, "allen_models")
hallermann_folder = join(cell_models_folder, "HallermannEtAl2012")

bbp_mod_folder = join(cell_models_folder, "bbp_mod")
os.makedirs(bbp_folder, exist_ok=True)


def download_BBP_model(cell_name="L5_TTPC2_cADpyr232_1"):

    print("Downloading BBP model: ", cell_name)
    url = "https://bbp.epfl.ch/nmc-portal/assets/documents/static/downloads-zip/{}.zip".format(cell_name)

    if sys.version < '3':
        from urllib2 import urlopen
    else:
        from urllib.request import urlopen
    import ssl
    import zipfile
    #get the model files:
    u = urlopen(url, context=ssl._create_unverified_context())

    localFile = open(join(bbp_folder, '{}.zip'.format(cell_name)), 'wb')
    localFile.write(u.read())
    localFile.close()
    #unzip:
    myzip = zipfile.ZipFile(join(bbp_folder, '{}.zip'.format(cell_name)), 'r')
    myzip.extractall(bbp_folder)
    myzip.close()
    os.remove(join(bbp_folder, '{}.zip'.format(cell_name)))
    #compile mod files every time, because of incompatibility with Mainen96 files:
    # mod_pth = join(hay_folder, "mod/")
    #
    # if "win32" in sys.platform:
    #     warn("no autompile of NMODL (.mod) files on Windows.\n"
    #          + "Run mknrndll from NEURON bash in the folder "
    #            "L5bPCmodelsEH/mod and rerun example script")
    #     if not mod_pth in neuron.nrn_dll_loaded:
    #         neuron.h.nrn_load_dll(join(mod_pth, "nrnmech.dll"))
    #     neuron.nrn_dll_loaded.append(mod_pth)
    # else:
    #     os.system('''
    #               cd {}
    #               nrnivmodl
    #               '''.format(mod_pth))
    #     neuron.load_mechanisms(mod_pth)

    # attempt to set up a folder with all unique mechanism mod files, compile, and
    # load them all
    compile_bbp_mechanisms(cell_name)


def download_allen_model(cell_name="473863035"):

    print("Downloading Allen model: ", cell_name)
    url = "https://api.brain-map.org/neuronal_model/download/%s" % cell_name

    cwd = os.getcwd()
    os.chdir(allen_folder)

    os.system("wget %s" % url)
    os.system("mv %s neuronal_model_%s.zip" % (cell_name, cell_name))

    os.chdir(cwd)

    import zipfile

    myzip = zipfile.ZipFile(join(allen_folder, 'neuronal_model_{}.zip'.format(cell_name)), 'r')
    myzip.extractall(join(allen_folder, 'neuronal_model_{}'.format(cell_name)))
    myzip.close()
    os.remove(join(allen_folder, 'neuronal_model_{}.zip'.format(cell_name)))

    mod_folder = join(allen_folder, 'neuronal_model_{}'.format(cell_name), "modfiles")
    if not os.path.isdir(join(mod_folder, "x86_64")):
        print("Compiling mechanisms ...")
        cwd = os.getcwd()
        os.chdir(mod_folder)
        os.system("nrnivmodl")
        os.chdir(cwd)


def download_hallermann_model():

    print("Downloading Hallermann model")
    if sys.version < '3':
        from urllib2 import urlopen
    else:
        from urllib.request import urlopen
    import ssl
    from warnings import warn
    import zipfile
    #get the model files:
    u = urlopen('https://senselab.med.yale.edu/modeldb/eavBinDown?o=144526&a=23&mime=application/zip',
                context=ssl._create_unverified_context())
    localFile = open(join(cell_models_folder, 'HallermannEtAl2012_r1.zip'), 'wb')
    localFile.write(u.read())
    localFile.close()
    #unzip:
    myzip = zipfile.ZipFile(join(cell_models_folder, 'HallermannEtAl2012_r1.zip'), 'r')
    myzip.extractall(cell_models_folder)
    myzip.close()

    # Remove NEURON GUI from model files:
    model_file_ = open(join(hallermann_folder, "Cell parameters.hoc"), 'r')
    new_lines = ""
    for line in model_file_:
        changes = line.replace('load_proc("nrn', '//load_proc("nrn')
        changes = changes.replace('load_file("nrn', '//load_file("nrn')
        new_lines += changes
    new_lines += "parameters()\ngeom_nseg()\ninit_channels()\n"

    model_file_.close()
    model_file_mod = open(join(hallermann_folder, "Cell parameters_mod.hoc"), 'w')
    model_file_mod.write(new_lines)
    model_file_mod.close()

    #compile mod files every time, because of incompatibility with Mainen96 files:
    mod_pth = join(hallermann_folder)

    if "win32" in sys.platform:
        warn("no autompile of NMODL (.mod) files on Windows.\n"
             + "Run mknrndll from NEURON bash in the folder "
               "L5bPCmodelsEH/mod and rerun example script")
        if not mod_pth in neuron.nrn_dll_loaded:
            neuron.h.nrn_load_dll(join(mod_pth, "nrnmech.dll"))
        neuron.nrn_dll_loaded.append(mod_pth)
    else:
        os.system('''
                  cd {}
                  nrnivmodl
                  '''.format(mod_pth))
        neuron.load_mechanisms(mod_pth)


def compile_bbp_mechanisms(cell_name):
    from warnings import warn

    if not os.path.isdir(bbp_mod_folder):
        os.mkdir(bbp_mod_folder)
    cell_folder = join(bbp_folder, cell_name)
    for nmodl in glob(join(cell_folder, 'mechanisms', '*.mod')):
        while not os.path.isfile(join(bbp_mod_folder, os.path.split(nmodl)[-1])):
            if "win32" in sys.platform:
                os.system("copy {} {}".format(nmodl, bbp_mod_folder))
            else:
                os.system('cp {} {}'.format(nmodl,
                                            join(bbp_mod_folder, '.')))
    CWD = os.getcwd()
    os.chdir(bbp_mod_folder)
    if "win32" in sys.platform:
        warn("no autompile of NMODL (.mod) files on Windows. " +
             "Run mknrndll from NEURON bash in the folder %s" % bbp_mod_folder +
             "and rerun example script")
    else:
        os.system('nrnivmodl')
    os.chdir(CWD)


def posixpth(pth):
    """
    Replace Windows path separators with posix style separators
    """
    return pth.replace(os.sep, posixpath.sep)


def get_templatename(f):
    '''
    Assess from hoc file the templatename being specified within

    Arguments
    ---------
    f : file, mode 'r'

    Returns
    -------
    templatename : str

    '''
    for line in f.readlines():
        if 'begintemplate' in line.split():
            templatename = line.split()[-1]
            continue
    return templatename


def return_BBP_neuron(cell_name, tstop, dt):

    # load some required neuron-interface files
    neuron.h.load_file("stdrun.hoc")
    neuron.h.load_file("import3d.hoc")

    CWD = os.getcwd()
    cell_folder = join(join(bbp_folder, cell_name))
    if not os.path.isdir(cell_folder):
        download_BBP_model(cell_name)

    if not hasattr(neuron.h, "CaDynamics_E2"):
        neuron.load_mechanisms(bbp_mod_folder)
    os.chdir(cell_folder)
    add_synapses = False
    # get the template name
    f = open("template.hoc", 'r')
    templatename = get_templatename(f)
    f.close()

    # get biophys template name
    f = open("biophysics.hoc", 'r')
    biophysics = get_templatename(f)
    f.close()

    # get morphology template name
    f = open("morphology.hoc", 'r')
    morphology = get_templatename(f)
    f.close()

    # get synapses template name
    f = open(posixpth(os.path.join("synapses", "synapses.hoc")), 'r')
    synapses = get_templatename(f)
    f.close()

    neuron.h.load_file('constants.hoc')

    if not hasattr(neuron.h, morphology):
        """Create the cell model"""
        # Load morphology
        neuron.h.load_file(1, "morphology.hoc")
    if not hasattr(neuron.h, biophysics):
        # Load biophysics
        neuron.h.load_file(1, "biophysics.hoc")
    if not hasattr(neuron.h, synapses):
        # load synapses
        neuron.h.load_file(1, posixpth(os.path.join('synapses', 'synapses.hoc')
                                       ))
    if not hasattr(neuron.h, templatename):
        # Load main cell template
        neuron.h.load_file(1, "template.hoc")

    templatefile = posixpth(os.path.join(cell_folder, 'template.hoc'))

    morphologyfile = glob(os.path.join('morphology', '*'))[0]


    # Instantiate the cell(s) using LFPy
    cell = LFPy.TemplateCell(morphology=morphologyfile,
                             templatefile=templatefile,
                             templatename=templatename,
                             templateargs=1 if add_synapses else 0,
                             tstop=tstop,
                             dt=dt,
                             nsegs_method=None)
    os.chdir(CWD)
    # set view as in most other examples
    cell.set_rotation(x=np.pi / 2)
    return cell


def download_hay_model():

    print("Downloading Hay model")
    if sys.version < '3':
        from urllib2 import urlopen
    else:
        from urllib.request import urlopen
    import ssl
    from warnings import warn
    import zipfile
    #get the model files:
    u = urlopen('http://senselab.med.yale.edu/ModelDB/eavBinDown.asp?o=139653&a=23&mime=application/zip',
                context=ssl._create_unverified_context())
    localFile = open(join(cell_models_folder, 'L5bPCmodelsEH.zip'), 'wb')
    localFile.write(u.read())
    localFile.close()
    #unzip:
    myzip = zipfile.ZipFile(join(cell_models_folder, 'L5bPCmodelsEH.zip'), 'r')
    myzip.extractall(cell_models_folder)
    myzip.close()

    #compile mod files every time, because of incompatibility with Mainen96 files:
    mod_pth = join(hay_folder, "mod/")

    if "win32" in sys.platform:
        warn("no autompile of NMODL (.mod) files on Windows.\n"
             + "Run mknrndll from NEURON bash in the folder "
               "L5bPCmodelsEH/mod and rerun example script")
        if not mod_pth in neuron.nrn_dll_loaded:
            neuron.h.nrn_load_dll(join(mod_pth, "nrnmech.dll"))
        neuron.nrn_dll_loaded.append(mod_pth)
    else:
        os.system('''
                  cd {}
                  nrnivmodl
                  '''.format(mod_pth))
        neuron.load_mechanisms(mod_pth)


def point_axon_down(cell):
    '''
    Make the axon of Hay model point downwards, start at soma mid point,
    comp for soma diameter

    Keyword arguments:
    :
        cell : LFPy.TemplateCell instance

    '''
    iaxon = cell.get_idx(section='axon')
    isoma = cell.get_idx(section='soma')
    cell.x[iaxon, 0] = cell.x[isoma].mean(axis=1)
    cell.x[iaxon, 1] = cell.x[isoma].mean(axis=1)

    cell.y[iaxon, 0] = cell.y[isoma].mean(axis=1)
    cell.y[iaxon, 1] = cell.y[isoma].mean(axis=1)

    j = 0
    for i in iaxon:
        cell.z[i, 0] = cell.z[isoma].mean(axis=1) \
                       - cell.d[isoma] / 2 - cell.length[i] * j
        cell.z[i, 1] = cell.z[isoma].mean(axis=1) \
                       - cell.d[isoma] / 2 - cell.length[i] - cell.length[i] * j
        j += 1

    ##point the pt3d axon as well
    for sec in cell.allseclist:
        if sec.name().rfind('axon') >= 0:
            x0 = cell.x[cell.get_idx(sec.name())[0], 0]
            y0 = cell.y[cell.get_idx(sec.name())[0], 0]
            z0 = cell.z[cell.get_idx(sec.name())[0], 0]
            L = sec.L
            for j in range(int(neuron.h.n3d(sec=sec))):
                neuron.h.pt3dchange(j, x0, y0, z0,
                                    sec.diam, sec=sec)
                z0 -= L / (neuron.h.n3d(sec=sec) - 1)

    # let NEURON know about the changes we just did:
    neuron.h.define_shape()
    cell.x3d, cell.y3d, cell.z3d, cell.diam3d, cell.arc3d = cell._collect_pt3d()


def return_hay_cell(tstop, dt):
    if not os.path.isfile(join(hay_folder, 'morphologies', 'cell1.asc')):
        download_hay_model()

    neuron.load_mechanisms(join(hay_folder, 'mod'))
    cell_params = {
        'morphology': join(hay_folder, "morphologies", "cell1.asc"),
        'templatefile': [join(hay_folder, 'models', 'L5PCbiophys3.hoc'),
                         join(hay_folder, 'models', 'L5PCtemplate.hoc')],
        'templatename': 'L5PCtemplate',
        'templateargs': join(hay_folder, 'morphologies', 'cell1.asc'),
        'passive': False,
        'nsegs_method': None,
        'dt': dt,
        'tstart': 0,
        'tstop': tstop,
        'v_init': -75,
        'celsius': 34,
        'pt3d': True,
    }

    #Initialize cell instance, using the LFPy.Cell class
    cell = LFPy.TemplateCell(**cell_params)
    cell.set_rotation(x=4.729, y=-3.166)
    point_axon_down(cell)
    return cell


def return_hallermann_cell(tstop, dt, ):

    #model_folder = join(cell_models_folder, 'HallermannEtAl2012')
    if not os.path.isfile(join(hallermann_folder, '28_04_10_num19.hoc')):
        download_hallermann_model()
    neuron.load_mechanisms(hallermann_folder)

    # Define cell parameters
    cell_parameters = {  # various cell parameters,
        'morphology': join(hallermann_folder, '28_04_10_num19.hoc'),
        'v_init': -80.,  # initial crossmembrane potential
        'passive': False,  # switch on passive mechs
        'nsegs_method': 'lambda_f',
        'lambda_f': 500.,
        'dt': dt,  # [ms] dt's should be in powers of 2 for both,
        'tstart': 0,  # start time of simulation, recorders start at t=0
        'tstop': tstop,
        "extracellular": False,
        "pt3d": True,
        'custom_code': [join(hallermann_folder, 'Cell parameters_mod.hoc'),
                        #join(hallermann_folder, 'charge.hoc')
                        ]
    }

    cell = LFPy.Cell(**cell_parameters)
    cell.set_rotation(x=np.pi/2, y=-0.1, z=0)
    return cell


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

    if not hasattr(neuron.h, "CaDynamics"):
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
        'nsegs_method': 'fixed_length',  # spatial discretization method
        'max_nsegs_length': 20.,
        #'lambda_f' : 200.,           # frequency where length constants are computed
        'dt': dt,      # simulation time step size
        'tstart': 0,      # start time of simulation, recorders start at t=0
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

def return_spike_time_idxs_imem(im):
    """Returns threshold crossings for membrane
    potential of single compartment"""
    # num_tsteps_in_half_ms = int(0.5 / self.dt)
    crossings = []
    stds = np.std(im, axis=1)
    max_std_comp_idx = np.argmax(stds)
    print(max_std_comp_idx)
    threshold = 3 * stds[max_std_comp_idx]

    # if np.max(im[max_std_comp_idx, :]) < threshold:
    #     return np.array([])
    for t_idx in range(1, len(im[max_std_comp_idx])):
        if np.abs(im[max_std_comp_idx, t_idx - 1]) < threshold <= im[max_std_comp_idx, t_idx]:
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


def return_spiketime_idx(cell):
    spike_window = [-1, 4]
    spike_time_idxs = return_spike_time_idxs(cell.somav)

    if cell.tvec[spike_time_idxs[-1]] + spike_window[1] <= cell.tvec[-1]:
        used_idx = spike_time_idxs[-1]
    elif len(spike_time_idxs) > 2:
        used_idx = spike_time_idxs[-2]
    else:
        used_idx = spike_time_idxs[-1]
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


def find_good_stim_amplitude_BBP(cell_name, dt, tstop):
    amp = -0.05#-0.05
    num_spikes = 0
    min_spikes = 2
    max_spikes = 10

    while not min_spikes <= num_spikes <= max_spikes:
        print("Testing amp {:1.3f} on cell {}".format(amp, cell_name))
        if num_spikes < min_spikes:
            amp *= 1.5
        elif num_spikes > max_spikes:
            amp *= 0.75
        cell = return_BBP_neuron(cell_name, dt, tstop)
        synapse, cell = insert_current_stimuli(cell, amp)
        cell.simulate(rec_vmem=True, rec_imem=True)

        num_spikes = len(return_spike_time_idxs(cell.somav))
        if not min_spikes <= num_spikes <= max_spikes:
            cell.__del__()
            cell = None
            synapse = None

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

    #model_ids = [f.split('_')[-1] for f in os.listdir(cell_models_folder)
    #              if f.startswith("neuronal_model_") and
    #              os.path.isdir(join(cell_models_folder, f))][::-1]

    #print(model_ids)

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
    #  ]
    # model_ids = [329321704,
    #      471087975,
    #      472299294,
    #      472300877,
    #      472306460,
    #      472363762,
    #      472451419,
    #      473834758,
    #      473862496,
    #      473863035,
    #      473863578,
    #      473871773,
    #      476630516,
    #      476637796,
    #      477876583,
    #      477880244,
    #      478045347,
    #      478047588,
    #      478047816,
    #      478513398,
    #      478809991,
    #      479427369,
    #      479427516,
    #      480051220,
    #      480361288,
    #      480624414,
    #      480630344,
    #      480633088,
    #      482529696,
    #      482657528,
    #      482934212,
    #      483108201,
    #      483109057,
    #      485507735,
    #      485513184,
    #      485591806,
    #      485720587,
    #      486052412,
    #      486508647,
    #      486509958,
    #      486558444,
    #      486909496,
    #      488083972,
    #      488462783,
    #      491766131,
    #      496490646,
    #      496538888,
    #      496930324,
    #      497229061,
    #      497229075,
    #      497229082,
    #      497229117,
    #      497229124,
    #      497229138,
    #      497232298,
    #      497232312,
    #      497232339,
    #      497232363,
    #      497232419,
    #      497232429,
    #      497232482,
    #      497232507,
    #      497232564,
    #      497232571,
    #      497232629,
    #      497232641,
    #      497232692,
    #      497232735,
    #      497232839,
    #      497232858,
    #      497232946,
    #      497232985,
    #      497232999,
    #      497233049,
    #      497233125,
    #      497233139,
    #      497233244,
    #      497233278,
    #      497233285,
    #      497233292,
    #      497233307,
    #      515175260,
    #      515175291,
    #      515175354,
    #      527109578]
    model_ids = [f[-9:] for f in os.listdir(allen_folder) if f.startswith("neuronal_model") and
                  os.path.isdir(join(allen_folder, f))]
    #print(model_ids)
    #model_ids = [f for f in model_ids if f.startswith("4972")]
    # dt = 2**-7
    sampling_rate = 30000  # Hz
    dt = 1 / sampling_rate * 1000 / 4
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

            model_folder = join(allen_folder, "neuronal_model_%s" % model_id)
            if not os.path.isdir(model_folder):
                download_allen_model(model_id)
            cell = return_allen_cell_model(model_folder, dt, tstop)
            eap_predictors = get_cell_spike_amp_tranfer_function(cell)

            model_type = cell.manifest["biophys"][0]["model_type"].split("-")[1]
            cell_type = cell.metadata["specimen"]["specimen_tags"][1]["name"].split("-")[1]
            cell_region = cell.metadata["specimen"]["structure"]["name"]
            cell_layer = cell.metadata["specimen"]["structure"]["name"].split(",")[1]

            #if "- spiny" in cell_type and "layer 1" not in cell_region:
            print("Running: ", model_id, model_type, cell_type, cell_layer)
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
                'x': np.array([cell.d[0]/2]),  # electrode requires 1d vector of positions
                'y': np.array([0]),
                'z': np.array([0]),
                "method": "root_as_point",
            }
            electrode = LFPy.RecExtElectrode(cell, **elec_params)
            eaps_hd = electrode.get_transformation_matrix() @ cell.imem * 1e3

            eap_predictors["id"] = model_id
            eap_predictors["model_type"] = model_type
            eap_predictors["cell_type"] = cell_type
            eap_predictors["cell_layer"] = cell_layer
            eap_predictors["eap_p2p"] = np.max(eaps_hd) - np.min(eaps_hd)
            eap_predictors["eap"] = eaps_hd
            eap_predictors["imem_p2p"] = np.max(cell.imem[0]) - np.min(cell.imem[0])
            eap_predictors["vmem_p2p"] = np.max(cell.vmem[0]) - np.min(cell.vmem[0])

            os.makedirs(data_folder, exist_ok=True)
            np.save(join(data_folder, "eap_predictors_%s.npy" % model_id), eap_predictors)

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


        # plot_idxs = [cell.somaidx[0],
        #              cell.get_closest_idx(z=-1e7),
        #              cell.get_closest_idx(z=1e7),]
        # idx_clr = {idx: ['b', 'cyan', 'orange', 'green', 'purple'][num]
        #            for num, idx in enumerate(plot_idxs)}


def plot_single_cell_detectable_volume(cell, fig_name, fig_folder, fig_suptitle=""):
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

    # save_neural_spike_data(cell, data_folder, model_id, eaps)

    plt.close("all")
    fig = plt.figure(figsize=[16, 9])
    # fig_suptitle = "%s; %s; %s; %s" % (model_id, model_type, cell_type, cell_region)
    fig.suptitle(fig_suptitle, size=14)

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
                       xticks=[], yticks=[], )

    ax6 = fig.add_axes([0.62, 0.01, 0.37, 0.9], frameon=False,
                       xticks=[], yticks=[],
                       xlim=[x0_ld - dz_ld / 4, x1_ld + dz_ld],
                       ylim=[z0_ld - dz_ld, z1_ld + dz_ld],
                       title="xz-plane",
                       aspect=1, )

    idx_apic = np.argmax(cell.z.mean(axis=1))
    idx_mid = cell.get_closest_idx(z=(np.max(cell.z) + np.min(cell.z)) / 2)
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
        'x': np.array([cell.d[0] / 2]),  # electrode requires 1d vector of positions
        'y': np.array([0]),
        'z': np.array([0]),
        "method": "root_as_point",
    }
    electrode = LFPy.RecExtElectrode(cell, **elec_params)
    eaps_hd = electrode.get_transformation_matrix() @ cell.imem * 1e3

    # eap_predictors["id"] = model_id
    # eap_predictors["model_type"] = model_type
    # eap_predictors["cell_type"] = cell_type
    # eap_predictors["cell_layer"] = cell_layer
    # eap_predictors["eap_p2p"] = np.max(eaps_hd) - np.min(eaps_hd)
    # eap_predictors["eap"] = eaps_hd
    # eap_predictors["imem_p2p"] = np.max(cell.imem[0]) - np.min(cell.imem[0])
    # eap_predictors["vmem_p2p"] = np.max(cell.vmem[0]) - np.min(cell.vmem[0])

    # os.makedirs(data_folder, exist_ok=True)
    # np.save(join(data_folder, "eap_predictors_%s.npy" % model_id), eap_predictors)

    ax3.text(1, np.min(eaps_hd[0]) / 2, r"$V_{\rm e}$")
    ax3.set_yticks([0])
    ax3.set_yticklabels(["0 µV"])
    ax3.plot([cell.tvec[-1], cell.tvec[-1]], [0, np.min(eaps_hd[0])], c='k', lw=3)
    ax3.text(cell.tvec[-1] + 2, np.min(eaps_hd[0]) / 2, "{:1.0f} µV".format(np.abs(np.min(eaps_hd[0]))))

    ax1.plot(elec_params["x"], elec_params["z"], 'D', c='r', ms=9)
    # ax4a.plot(elec_params["x"], elec_params["z"], 'D', c='r', ms=9)

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
    ax3.axvline(spike_time_idx * cell.dt, c='gray', lw=0.5, ls='--')
    ax2.axvline(spike_time_idx * cell.dt, c='gray', lw=0.5, ls='--')

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
    plt.savefig(join(fig_folder, "spike_%s.png" % fig_name), dpi=150)


def plot_cell_secs(cell, ax_m_side, ax_m_top):

    possible_names = ["Myelin", "axon", "Unmyelin", "Node", "node", "my",
                      "hilloc",
                      "hill", "apic", "dend", "soma"]

    sec_clrs = {"Myelin": 'olive',
            "dend": '0.6',
            "soma": '0.0',
            'apic': '0.8',
            "axon": 'lightgreen',
            "node": 'r',
            "my": '0.5',
            "Unmyelin": 'salmon',
            "Node": 'r',
            "hilloc": 'lightblue',
            "hill": 'pink',}

    legend_dict = {"soma": "soma",
                   "my": "myelin",
                   "node": "node of Ranvier",
                   "axon": "AIS",
                   "dend": "dend",
                   "apic": "apic"}

    used_clrs = []
    cell_clr_list = []
    for idx in range(len(cell.x)):
        sec_name = cell.get_idx_name(idx)[1]

        if "node" in sec_name:
            zorder = 5000
        else:
            zorder = -50

        for ax_name in possible_names:
            if ax_name in sec_name:

                c = sec_clrs[ax_name]
                if not ax_name in used_clrs:
                    used_clrs.append(ax_name)
        cell_clr_list.append(c)
        ax_m_top.plot(cell.x[idx], cell.y[idx], '-',
              c=c, clip_on=False, lw=np.sqrt(cell.d[idx]) * 2, zorder=zorder)
        ax_m_side.plot(cell.x[idx], cell.z[idx], '-',
              c=c, clip_on=True, lw=np.sqrt(cell.d[idx]) * 2,
                       zorder=zorder)
        if "node" in sec_name:
            ax_m_top.plot(cell.x[idx].mean(), cell.y[idx].mean(), 'o', ms=3,
                          c=c)
            ax_m_side.plot(cell.x[idx].mean(), cell.z[idx].mean(), 'o', ms=3,
                          c=c)

    lines = []
    line_names = []
    for name in used_clrs:
        l, = ax_m_side.plot([0], [0], lw=2, c=sec_clrs[name])
        #if not "soma" in name:
        lines.append(l)
        line_names.append(legend_dict[name])
    ax_m_side.legend(lines, line_names, frameon=False, fontsize=10,
                    loc=(-0.45, -0.15), ncol=1)
    return cell_clr_list


def plot_superimposed_sec_type(cell, ax):

    possible_names = ["Myelin", "axon", "Unmyelin", "Node", "node", "my",
                      "hilloc",
                      "hill", "apic", "dend", "soma"]
    axon_names = ["Myelin", "axon", "Unmyelin", "Node", "node", "my",
                      "hilloc",
                      "hill"]
    sec_clrs = {"Myelin": 'olive',
            "dend": '0.6',
            "soma": '0.0',
            'apic': '0.8',
            "axon": 'lightgreen',
            "node": 'r',
            "my": 'olive',
            "Unmyelin": 'lightgreen',
            "Node": 'r',
            "hilloc": 'lightblue',
            "hill": 'pink',}

    legend_dict = {"soma": "soma",
                   "my": "myelin",
                   "node": "node of Ranvier",
                   "axon": "AIS",
                   "dend": "dend",
                   "apic": "apic"}

    used_clrs = []
    cell_clr_list = []
    for idx in range(len(cell.x)):
        sec_name = cell.get_idx_name(idx)[1]
        # if "node" in sec_name:
        #     zorder = 5000
        # else:
        zorder = 9

        for ax_name in axon_names:
            if ax_name in sec_name:
                c = sec_clrs[ax_name]
                if not ax_name in used_clrs:
                    used_clrs.append(ax_name)
                cell_clr_list.append(c)

                ax.plot(cell.x[idx], cell.z[idx], '-',
                      c=c, clip_on=True, lw=1, zorder=zorder)
                if "node" in sec_name:
                    ax.plot(cell.x[idx].mean(), cell.z[idx].mean(), 'o', ms=3, c=c)

    # lines = []
    # line_names = []
    # for name in used_clrs:
    #     l, = ax.plot([0], [0], lw=2, c=sec_clrs[name])
    #     #if not "soma" in name:
    #     lines.append(l)
    #     line_names.append(legend_dict[name])
    # ax.legend(lines, line_names, frameon=False, fontsize=10,
    #                 loc=(-0.45, -0.15), ncol=1)
    return cell_clr_list


def get_cell_spike_amp_tranfer_function(cell):
    somasec = None
    for sec in neuron.h.allsec():
        if "soma" in sec.name():
            somasec = sec
            break
    #print(dir(somasec))
    soma_children = somasec.children()
    child_diams = np.zeros(len(soma_children))
    for c_idx, childsec in enumerate(soma_children):
        child_diams[c_idx] = childsec.diam
    child_diams_23 = np.sum(child_diams**(3 / 2))
    T_max = child_diams_23 / (somasec.diam / 2) * np.sqrt(
        somasec.cm / somasec.Ra)
    eap_predictors = {"soma_diam": somasec.diam,
                      "child_diams_23": child_diams_23,
                      "soma_cm": somasec.cm,
                      "soma_Ra": somasec.Ra,
                      "T_max": T_max,
                      }
    return eap_predictors


def inspect_cells():

    model_ids = [f.split('_')[-1] for f in os.listdir(cell_models_folder)
                  if f.startswith("neuronal_model_") and
                  os.path.isdir(join(cell_models_folder, f))][::-1]

    print(model_ids)

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

            print("Running: ", model_id, model_type, cell_type, cell_region)
            plt.close("all")
            fig = plt.figure(figsize=[16, 9])
            fig.suptitle("%s; %s; %s; %s" % (model_id, model_type, cell_type, cell_region))
            ax_m_side = fig.add_axes([0.1, 0.08, 0.12, 0.63], aspect=1)
            ax_m_top = fig.add_axes([0.1, 0.77, 0.12, 0.20], aspect=1)
            ax_diam = fig.add_axes([0.27, 0.1, 0.12, 0.80], xlim=[0, 3],
                                   xlabel="diameter (µm)", ylabel="z (µm)",
                                   title="diamter (soma={:1.1f} µm)".format(cell.d[0]))
            ax_ions = fig.add_axes([0.44, 0.1, 0.12, 0.80],
                                   xlabel="gbar (S/cm²)", ylabel="z (µm)",
                                   title="ion-channel conductance")
            ax_cm = fig.add_axes([0.61, 0.1, 0.12, 0.80],
                                   xlabel="Cm (µF/cm²)", ylabel="z (µm)",
                                   title="membrane capacitance")

            ax_ra = fig.add_axes([0.78, 0.1, 0.12, 0.80],
                                   xlabel="Ra (Ohm m)", ylabel="z (µm)",
                                   title="intracellular resistance")

            cell_clr_list = plot_cell_secs(cell, ax_m_side, ax_m_top)
            #print(np.max(cell.d) + 1)
            #ax_m.plot(cell.x.T, cell.z.T, )
            for idx in range(cell.totnsegs):
                ax_diam.plot(cell.d[idx], cell.z[idx].mean(), 'o', c=cell_clr_list[idx])

            #idx = 0

            ion_names = [#"Ca_HVA", "Ca_LVA", "K_T", "Kd", "SK", "Kv2like", "Kv3_1",
                         "Nap", "NaTs", "NaTa", "NaV"]

            ion_gbars = {}
            lines = []
            line_names = []

            i = 0
            ion_gbars["g_pas"] = np.zeros(cell.totnsegs)
            ion_gbars["c_m"] = np.zeros(cell.totnsegs)
            ion_gbars["ra"] = np.zeros(cell.totnsegs)
            for sec in neuron.h.allsec():
                for seg in sec:
                    ion_gbars["g_pas"][i] = seg.pas.g
                    ion_gbars["ra"][i] = sec.Ra
                    ion_gbars["c_m"][i] = seg.cm
                    i += 1

            ax_cm.plot(ion_gbars["c_m"], cell.z.mean(axis=-1), 'o', ms=5)
            ax_ra.plot(ion_gbars["ra"], cell.z.mean(axis=-1), 'o', ms=5)

            li, = ax_ions.plot(ion_gbars["g_pas"], cell.z.mean(axis=-1), 'o', ms=5)
            lines.append(li)
            line_names.append("g_pas")

            for ion_name in ion_names:
                ion_gbars[ion_name] = np.zeros(cell.totnsegs)
                i = 0
                for sec in neuron.h.allsec():
                    for seg in sec:
                        if hasattr(seg, ion_name):
                            ion_gbars[ion_name][i] = eval("seg.%s.gbar" % ion_name)

                        i += 1
                    # for k in dir(seg):
                    #     print(k)
                    #     if hasattr(seg, "%s."
                    #     if k.startswith("gbar"):
                    #         print(k)

#                print(sec.name())
                li, = ax_ions.plot(ion_gbars[ion_name], cell.z.mean(axis=-1), 'o', ms=4)
                lines.append(li)
                line_names.append(ion_name)
            ax_ions.legend(lines, line_names)

            T_max = get_cell_spike_amp_tranfer_function(cell)

            fig.savefig(join(fig_folder, "inspection_%s.png" % model_id), dpi=150)
            cell.__del__()
            os._exit(0)
        else:
            os.waitpid(pid, 0)

        #import sys; sys.exit()


def analyze_eap_amplitudes():
    data_folder = join("..", "model_scan", "sim_data")
    filelist = [f for f in os.listdir(data_folder) if f.startswith("eap_predictor")]

    num_cells = len(filelist)
    eap_p2p = np.zeros(num_cells)
    imem_p2p = np.zeros(num_cells)
    vmem_p2p = np.zeros(num_cells)
    child_diams = np.zeros(num_cells)
    soma_diams = np.zeros(num_cells)
    transfunc = np.zeros(num_cells)
    soma_cm = np.zeros(num_cells)
    soma_Ra = np.zeros(num_cells)
    cell_clr = lambda idx: plt.cm.rainbow(idx / (num_cells - 1))

    for idx, f in enumerate(filelist):
        pred_dict = np.load(join(data_folder, f), allow_pickle=True)[()]
        eap_p2p[idx] = pred_dict['eap_p2p']
        transfunc[idx] = pred_dict['T_max']
        imem_p2p[idx] = pred_dict['imem_p2p']
        vmem_p2p[idx] = pred_dict['vmem_p2p']
        child_diams[idx] = pred_dict['child_diams_23']
        soma_diams[idx] = pred_dict['soma_diam']
        soma_cm[idx] = pred_dict['soma_cm']
        soma_Ra[idx] = pred_dict['soma_Ra']

    ordered_idxs = np.argsort(eap_p2p)
    print(filelist[ordered_idxs[0]], filelist[ordered_idxs[-1]])
    eap_p2p = eap_p2p[ordered_idxs]
    transfunc = transfunc[ordered_idxs]
    imem_p2p = imem_p2p[ordered_idxs]
    vmem_p2p = vmem_p2p[ordered_idxs]
    child_diams = child_diams[ordered_idxs]
    soma_diams = soma_diams[ordered_idxs]
    soma_cm = soma_cm[ordered_idxs]

    fig = plt.figure(figsize=[22, 15])
    fig.subplots_adjust(bottom=0.05, top=0.98)
    num_rows = 8
    num_cols = 1
    ax1 = fig.add_subplot(num_rows, num_cols, 1, ylabel="EAP p2p\n(µV)", xticks=[])
    ax2 = fig.add_subplot(num_rows, num_cols, 2, ylabel="soma $I_m$\n(nA)", xticks=[])
    ax3 = fig.add_subplot(num_rows, num_cols, 3, ylabel="soma $V_m$\n(mV)", xticks=[])
    ax4 = fig.add_subplot(num_rows, num_cols, 4, ylabel="T_max", xticks=[])
    ax5 = fig.add_subplot(num_rows, num_cols, 5, ylabel="(d_dend)$^{(3/2)}$", xticks=[])
    ax6 = fig.add_subplot(num_rows, num_cols, 6, ylabel="soma diam\n(µm)", xticks=[])
    ax7 = fig.add_subplot(num_rows, num_cols, 7, ylabel="soma cm\n(µF/cm²)", xticks=[])
    ax8 = fig.add_subplot(num_rows, num_cols, 8, ylabel="soma Ra\n(Ohm cm)", xticks=[])

    ax8.axhline(150, lw=0.5, ls='--')

    for idx in range(num_cells):
        ax1.plot(idx, eap_p2p[idx], 'o', c=cell_clr(idx))
        ax2.plot(idx, imem_p2p[idx], 'o', c=cell_clr(idx))
        ax3.plot(idx, vmem_p2p[idx], 'o', c=cell_clr(idx))
        ax4.plot(idx, transfunc[idx], 'o', c=cell_clr(idx))
        ax5.plot(idx, child_diams[idx], 'o', c=cell_clr(idx))
        ax6.plot(idx, soma_diams[idx], 'o', c=cell_clr(idx))
        ax7.plot(idx, soma_cm[idx], 'o', c=cell_clr(idx))
        ax8.plot(idx, soma_Ra[idx], 'o', c=cell_clr(idx))

    fig.savefig("EAP_amp_summary.png", dpi=100)


def recreate_allen_data():

    import analyse_exp_data as axd

    model_ids = [f.split('_')[-1] for f in os.listdir(allen_folder)
                  if f.startswith("neuronal_model_") and
                  os.path.isdir(join(allen_folder, f))]

    print(model_ids)
    # dt = 2**-5
    sampling_rate = 30000  # Hz
    dt = 1 / sampling_rate * 1000 # ???
    #tstop = 120
    data_folder = join("..", "model_scan", "sim_data")
    fig_folder = join("..", "exp_data", "simulated", "allen")
    num_trials = 100
    os.makedirs(fig_folder, exist_ok=True)
    data_folder = join("..", "exp_data", "NPUltraWaveforms")

    elecs_x = np.load(join(data_folder, "channels.xcoords.npy"))[:, 0]
    elecs_z = np.load(join(data_folder, "channels.ycoords.npy"))[:, 0]

    elec_params = {
        'sigma': sigma,  # extracellular conductivity
        'x': elecs_x,  # electrode requires 1d vector of positions
        'y': np.zeros(len(elecs_x)),
        'z': elecs_z,
        'r': 2.5,
        'n': 20,
        'N': [0, 1, 0],
        "method": "root_as_point",
    }
    waveform_collection = []
    soma_distance_from_plane = []
    cell_type_list = []
    counter = 0
    for model_id in model_ids:

        # pid = os.fork()
        # if pid == 0:

        model_folder = join(cell_models_folder, "allen_models", "neuronal_model_%s" % model_id)
        cell = return_allen_cell_model(model_folder, dt, 120)

        model_type = cell.manifest["biophys"][0]["model_type"].split("-")[1]
        cell_type = cell.metadata["specimen"]["specimen_tags"][1]["name"].split("-")[1]
        cell_region = cell.metadata["specimen"]["structure"]["name"]
        cell_layer = cell.metadata["specimen"]["structure"]["name"].split(",")[1]

        cell_name = model_type + cell_type + ' ' + cell_region

        try:
            cell.imem = np.load(os.path.join(imem_eap_folder, "imem_filt2_%s.npy" % model_id))[:, ::4]
        except FileNotFoundError:
            print("Did not find: ", model_id)
            continue
        cell.tvec = np.arange(len(cell.imem[0, :])) * dt

        #if "- spiny" in cell_type and "layer 1" not in cell_region:
        print("Running: ", cell_name)
        # model_folder = join(cell_models_folder,  "neuronal_model_{}".format(model_id))
        #cell.__del__()
        #cell = find_good_stim_amplitude_allen(model_id, model_folder, dt, tstop)

        for trial_idx in range(num_trials):
            np.random.seed(int(model_id) + trial_idx)
            #spike_time_idx = extract_spike(cell)

            cell.set_rotation(x=np.random.uniform(0, 2 * np.pi) / 24,
                              y=np.random.uniform(0, 2 * np.pi) / 24,
                              z=np.random.uniform(0, 2 * np.pi))

            cell.set_pos(x=np.random.uniform(-axd.dx * 3, np.max(axd.x) + axd.dx * 3),
                         y=np.random.uniform(-60, -0),
                         z=np.random.uniform(-axd.dz, np.max(axd.z) + axd.dz))

            electrode = LFPy.RecExtElectrode(cell, **elec_params)
            eaps = electrode.get_transformation_matrix() @ cell.imem * 1e3

            eap_ = eaps.T
            t_ = cell.tvec
            if np.max(np.abs(eap_)) > 30:
                fig_name = "sim_allen_mouse_%s_%d_prefilt_downsampled_%d" % (model_id, trial_idx, counter)
                axd.plot_NPUltraWaveform(eap_, t_, fig_name,
                                         fig_folder, cell)
                waveform_collection.append(eap_)
                cell_type_list.append(cell_name)
                soma_distance_from_plane.append(cell.y[0].mean())
                counter += 1

                # os._exit(0)
            # else:
            #     os.waitpid(pid, 0)
    np.save(join(fig_folder, "..", "waveforms_sim_allen.npy"), waveform_collection)
    np.save(join(fig_folder, "..", "waveforms_sim_allen_soma_distance.npy"), soma_distance_from_plane)
    np.save(join(fig_folder, "..", "waveforms_sim_allen_celltype_list.npy"), cell_type_list)


def recreate_allen_data_hay():

    import analyse_exp_data as axd
    sampling_rate = 30000  # Hz
    dt = 1 / sampling_rate * 1000
    # dt = 2**-5
    # tstop = 120
    # filt_dict_high_pass = {'highpass_freq': 300,
    #                        'lowpass_freq': None,
    #                        'order': 1,
    #                        'filter_function': 'filtfilt',
    #                        'fs': 1 / (dt / 1000),
    #                        'axis': -1
    #                        }
    cell_name = "hay"
    fig_folder = join("..", "exp_data", "simulated", cell_name)
    os.makedirs(fig_folder, exist_ok=True)
    num_trials = 500
    cell = return_hay_cell(120, dt)
    #synapse, cell = insert_current_stimuli(cell, -0.4)
    #cell.simulate(rec_vmem=True, rec_imem=True)
    cell.imem = np.load(os.path.join(imem_eap_folder, "imem_filt2_%s.npy" % cell_name))[:, ::4]
    cell.tvec = np.arange(len(cell.imem[0, :])) * dt

    # print(cell.tvec, len(cell.tvec))
    #print(np.max(cell.somav))
    #spiketime_idx = return_spiketime_idx(cell)
    #t_window = [spiketime_idx - int(1 / dt), spiketime_idx + int(1.7 / dt)]
    waveform_collection = []
    soma_distance_from_plane = []
    data_folder = join("..", "exp_data", "NPUltraWaveforms")
    elecs_x = np.load(join(data_folder, "channels.xcoords.npy"))[:, 0]
    elecs_z = np.load(join(data_folder, "channels.ycoords.npy"))[:, 0]

    elec_params = {
        'sigma': sigma,  # Saline bath conductivity
        'x': elecs_x,  # electrode requires 1d vector of positions
        'y': np.zeros(len(elecs_x)),
        'z': elecs_z,
        'r': 2.5,
        'n': 20,
        'N': [0, 1, 0],
        "method": "root_as_point",
    }
    counter = 0
    for trial_idx in range(num_trials):

        #pid = os.fork()
        #if pid == 0:

        np.random.seed(12345 + trial_idx)

        # NOTE: THIS IS CUMULATIVE, so cell eventually will have all kinds of rotations
        cell.set_rotation(x=np.random.uniform(0, 2 * np.pi) / 24,
                          y=np.random.uniform(0, 2 * np.pi) / 24,
                          z=np.random.uniform(0, 2 * np.pi))

        cell.set_pos(x=np.random.uniform(-axd.dx * 3, np.max(axd.x) + axd.dx * 3),
                     y=np.random.uniform(-60, -0),
                     z=np.random.uniform(-axd.dz, np.max(axd.z) + axd.dz))

        electrode = LFPy.RecExtElectrode(cell, **elec_params)
        eaps = electrode.get_transformation_matrix() @ cell.imem * 1e3
        #eaps = elephant.signal_processing.butter(eaps, **filt_dict_high_pass)

        t_ = cell.tvec#cell.tvec[t_window[0]:t_window[1]] - cell.tvec[t_window[0]]
        eap_ = eaps.T#eaps[:, t_window[0]:t_window[1]].T
        if np.max(np.abs(eap_)) > 30:
            fig_name = "sim_%s_%s_prefilt_downsampled" % (cell_name, trial_idx)
            axd.plot_NPUltraWaveform(eap_, t_, fig_name,
                                     fig_folder, cell)
            waveform_collection.append(eap_)
            soma_distance_from_plane.append(cell.y[0].mean())
            counter += 1
       #     os._exit(0)
       # else:
       #     os.waitpid(pid, 0)
    np.save(join(fig_folder, "..", "waveforms_sim_hay.npy"), waveform_collection)
    np.save(join(fig_folder, "..", "waveforms_sim_hay_soma_distance.npy"), soma_distance_from_plane)


def recreate_allen_data_hallermann():

    cell_name = "hallermann"
    import analyse_exp_data as axd
    sampling_rate = 30000  # Hz
    dt = 1 / sampling_rate * 1000   # ???

    # tstop = 120
    # filt_dict_high_pass = {'highpass_freq': 300,
    #                        'lowpass_freq': None,
    #                        'order': 1,
    #                        'filter_function': 'filtfilt',
    #                        'fs': 1 / (dt / 1000),
    #                        'axis': -1
    #                        }
    fig_folder = join("..", "exp_data", "simulated", cell_name)
    os.makedirs(fig_folder, exist_ok=True)
    num_trials = 500
    cell = return_hallermann_cell(120, dt)
    #synapse, cell = insert_current_stimuli(cell, -0.4)
    #cell.simulate(rec_vmem=True, rec_imem=True)
    cell.imem = np.load(os.path.join(imem_eap_folder, "imem_filt2_%s.npy" % cell_name))[:, ::4]
    cell.tvec = np.arange(len(cell.imem[0, :])) * dt
    #print(np.max(cell.somav))
    #spiketime_idx = return_spiketime_idx(cell)
    #t_window = [spiketime_idx - int(1 / dt), spiketime_idx + int(1.7 / dt)]
    waveform_collection = []
    soma_distance_from_plane = []
    data_folder = join("..", "exp_data", "NPUltraWaveforms")
    elecs_x = np.load(join(data_folder, "channels.xcoords.npy"))[:, 0]
    elecs_z = np.load(join(data_folder, "channels.ycoords.npy"))[:, 0]

    elec_params = {
        'sigma': sigma,  # Saline bath conductivity
        'x': elecs_x,  # electrode requires 1d vector of positions
        'y': np.zeros(len(elecs_x)),
        'z': elecs_z,
        'r': 2.5,
        'n': 20,
        'N': [0, 1, 0],
        "method": "root_as_point",
    }
    counter = 0
    for trial_idx in range(num_trials):

        #pid = os.fork()
        #if pid == 0:

        np.random.seed(12345 + trial_idx)
        cell.set_rotation(x=np.random.uniform(0, 2 * np.pi) / 24,
                          y=np.random.uniform(0, 2 * np.pi) / 24,
                          z=np.random.uniform(0, 2 * np.pi))

        cell.set_pos(x=np.random.uniform(-axd.dx * 10, np.max(axd.x) + axd.dx * 10),
                     y=np.random.uniform(-200, -0),
                     z=np.random.uniform(-axd.dz * 10, np.max(axd.z) + axd.dz * 10))

        electrode = LFPy.RecExtElectrode(cell, **elec_params)
        eaps = electrode.get_transformation_matrix() @ cell.imem * 1e3
        #eaps = elephant.signal_processing.butter(eaps, **filt_dict_high_pass)


        t_ = cell.tvec#cell.tvec[t_window[0]:t_window[1]] - cell.tvec[t_window[0]]
        eap_ = eaps.T#eaps[:, t_window[0]:t_window[1]].T
        if np.max(np.abs(eap_)) > 30:
            fig_name = "sim_%s_%s_prefilt_downsampled" % (cell_name, counter)
            axd.plot_NPUltraWaveform(eap_, t_, fig_name,
                                     fig_folder, cell)
            waveform_collection.append(eap_)
            soma_distance_from_plane.append(cell.y[0].mean())

            counter += 1
       #     os._exit(0)
       # else:
       #     os.waitpid(pid, 0)
    np.save(join(fig_folder, "..", "waveforms_sim_%s.npy" % cell_name), waveform_collection)
    np.save(join(fig_folder, "..", "waveforms_sim_hallermann_soma_distance.npy"), soma_distance_from_plane)


def recreate_allen_data_axon():

    axon_type = "unmyelinated"
    cell_name = "%s_axon" % axon_type

    import analyse_exp_data as axd

    dt = 2**-5
    # tstop = 120
    # filt_dict_high_pass = {'highpass_freq': 300,
    #                        'lowpass_freq': None,
    #                        'order': 1,
    #                        'filter_function': 'filtfilt',
    #                        'fs': 1 / (dt / 1000),
    #                        'axis': -1
    #                        }
    fig_folder = join("..", "exp_data", "simulated", cell_name)
    os.makedirs(fig_folder, exist_ok=True)
    num_trials = 500
    from ECSbook_simcode import hallermann_axon_model as ha_ax
    if axon_type == "unmyelinated":
        cell = ha_ax.return_constructed_unmyelinated_axon(dt, 120, 0, 1)
    elif axon_type == "myelinated":
        cell = ha_ax.return_constructed_myelinated_axon(dt, 120, 0, 1)
    else:
        raise RuntimeError("axon_type not recognized")

    cell.imem = np.load(os.path.join(imem_eap_folder, "imem_filt2_%s.npy" % cell_name))[:, ::4]
    cell.tvec = np.arange(len(cell.imem[0, :])) * dt

    waveform_collection = []
    data_folder = join("..", "exp_data", "NPUltraWaveforms")
    elecs_x = np.load(join(data_folder, "channels.xcoords.npy"))[:, 0]
    elecs_z = np.load(join(data_folder, "channels.ycoords.npy"))[:, 0]


    elec_params = {
        'sigma': sigma,  # Saline bath conductivity
        'x': elecs_x,  # electrode requires 1d vector of positions
        'y': np.zeros(len(elecs_x)),
        'z': elecs_z,
        "method": "root_as_point",
    }
    counter = 0
    for trial_idx in range(num_trials):

        #pid = os.fork()
        #if pid == 0:

        np.random.seed(12345 + trial_idx)
        cell.set_rotation(x=np.random.uniform(0, 2 * np.pi) / 24,
                          y=np.random.uniform(0, 2 * np.pi) / 24,
                          z=np.random.uniform(0, 2 * np.pi))

        cell.set_pos(x=np.random.uniform(-axd.dx * 0.5, np.max(axd.x) + axd.dx * 0.5),
                     y=np.random.uniform(-5, -0),
                     z=np.random.uniform(-axd.dz * 0.5, np.max(axd.z) + axd.dz * 0.5))

        electrode = LFPy.RecExtElectrode(cell, **elec_params)
        eaps = electrode.get_transformation_matrix() @ cell.imem * 1e3
        #eaps = elephant.signal_processing.butter(eaps, **filt_dict_high_pass)


        t_ = cell.tvec#cell.tvec[t_window[0]:t_window[1]] - cell.tvec[t_window[0]]
        eap_ = eaps.T#eaps[:, t_window[0]:t_window[1]].T
        if np.max(np.abs(eap_)) > 5:
            fig_name = "sim_%s_%s_prefilt_downsampled" % (cell_name, counter)
            axd.plot_NPUltraWaveform(eap_, t_, fig_name,
                                     fig_folder, cell)
            waveform_collection.append(eap_)
            counter += 1
       #     os._exit(0)
       # else:
       #     os.waitpid(pid, 0)
    np.save(join(fig_folder, "..", "waveforms_sim_%s.npy" % cell_name), waveform_collection)


def recreate_allen_data_BBP():

    import analyse_exp_data as axd

    sampling_rate = 30000  # Hz
    dt = 1 / sampling_rate * 1000  # ???
    # dt = 2**-5  # Will only use every 4th time step of the original waveform (2**-7), see below
    #tstop = 120

    fig_folder = join("..", "exp_data", "simulated", "bbp")
    os.makedirs(fig_folder, exist_ok=True)
    num_trials = 50

    # neurons = ["L5_TTPC2_cADpyr232_2",
    #            "L5_MC_bAC217_1",
    #            "L5_NGC_bNAC219_5",
    #            ]
    # filt_dict_high_pass = {'highpass_freq': 300,
    #                        'lowpass_freq': None,
    #                        'order': 4,
    #                        'filter_function': 'filtfilt',
    #                        'fs': 1 / (dt / 1000),
    #                        'axis': -1
    #                        }
    cell_names = os.listdir(bbp_folder)
    data_folder = join("..", "exp_data", "NPUltraWaveforms")

    elecs_x = np.load(join(data_folder, "channels.xcoords.npy"))[:, 0]
    elecs_z = np.load(join(data_folder, "channels.ycoords.npy"))[:, 0]

    elec_params = {
        'sigma': sigma,  # Saline bath conductivity
        'x': elecs_x,  # electrode requires 1d vector of positions
        'y': np.zeros(len(elecs_x)),
        'z': elecs_z,
        'r': 2.5,
        'n': 20,
        'N': [0, 1, 0],
        "method": "root_as_point",
    }
    waveform_collection = []
    soma_distance_from_plane = []
    cell_type_list = []
    counter = 0
    for c_idx, cell_name in enumerate(cell_names):

        print(cell_name, c_idx, " / ", len(cell_names))
        # pid = os.fork()
        # if pid == 0:
        #cell = find_good_stim_amplitude_BBP(cell_name, dt, tstop)
        cell = return_BBP_neuron(cell_name, 120, dt)
        cell.imem = np.load(os.path.join(imem_eap_folder, "imem_filt2_%s.npy" % cell_name))[:, ::4]
        cell.tvec = np.arange(len(cell.imem[0, :])) * dt
        #synapse, cell = insert_current_stimuli(cell, -0.2)
        # cell.simulate(rec_vmem=True, rec_imem=True)
        #spike_time_idx = extract_spike(cell)

        #spiketime_idx = return_spiketime_idx(cell)
        #t_window = [spiketime_idx - int(1 / dt), spiketime_idx + int(1.7 / dt)]
        for trial_idx in range(num_trials):
            #if np.max(cell.somav) < -10:
            #    print("%s needs more input!" % cell_name)
            #    break

            np.random.seed(12345 + trial_idx + num_trials * c_idx)
            cell.set_rotation(x=np.random.uniform(0, 2 * np.pi) / 24,
                              y=np.random.uniform(0, 2 * np.pi) / 24,
                              z=np.random.uniform(0, 2 * np.pi))
            # cell.set_rotation(z=np.random.uniform(0, 2 * np.pi))
            # cell.set_pos(x=np.random.uniform(-axd.dx, np.max(axd.x) + axd.dx),
            #              y=np.random.uniform(-50, -10),
            #              z=np.random.uniform(-axd.dz, np.max(axd.z) + axd.dz))
            cell.set_pos(x=np.random.uniform(-axd.dx * 3, np.max(axd.x) + axd.dx * 3),
                         y=np.random.uniform(-60, -0),
                         z=np.random.uniform(-axd.dz, np.max(axd.z) + axd.dz))

            electrode = LFPy.RecExtElectrode(cell, **elec_params)
            eaps = electrode.get_transformation_matrix() @ cell.imem * 1e3

            #eaps = elephant.signal_processing.butter(eaps, **filt_dict_high_pass)
            t_ = cell.tvec #cell.tvec[t_window[0]:t_window[1]] - cell.tvec[t_window[0]]
            eap_ = eaps.T#[:, t_window[0]:t_window[1]]
            if np.max(np.abs(eap_)) > 30:
                fig_name = "sim_BBP_%s_%d_prefilt_downsampled_%d" % (cell_name, trial_idx, counter)
                # axd.plot_NPUltraWaveform(eap_, t_, fig_name,
                #                          fig_folder, cell)
                waveform_collection.append(eap_)
                cell_type_list.append(cell_name)
                soma_distance_from_plane.append(cell.y[0].mean())
                counter += 1

            # os._exit(0)
        # else:
        #     os.waitpid(pid, 0)
    np.save(join(fig_folder, "..", "waveforms_sim_BBP.npy"), waveform_collection)
    np.save(join(fig_folder, "..", "waveforms_sim_BBP_soma_distance.npy"), soma_distance_from_plane)
    np.save(join(fig_folder, "..", "waveforms_sim_BBP_celltype_list.npy"), cell_type_list)


def insert_synapses(cell, synparams, section, n, netstimParameters):
    """ Find n compartments to insert synapses onto """
    idx = cell.get_rand_idx_area_norm(section=section, nidx=n)

    #Insert synapses in an iterative fashion
    for i in idx:
        synparams.update({'idx': int(i)})
        # Create synapse(s) and setting times using the Synapse class in LFPy
        s = LFPy.Synapse(cell, **synparams)
        s.set_spike_times_w_netstim(**netstimParameters)


def insert_distributed_synaptic_input(cell, weight_scale):
    # Synaptic parameters taken from Hendrickson et al 2011
    # Excitatory synapse parameters:
    syn_params_AMPA = {
        'e': 0,  # reversal potential
        'syntype': 'Exp2Syn',  # conductance based exponential synapse
        'tau1': 1.,  # Time constant, rise
        'tau2': 3.,  # Time constant, decay
        'weight': 0.003 * weight_scale,  # Synaptic weight
        'record_current': False,  # record synaptic currents
    }
    # Excitatory synapse parameters
    syn_params_NMDA = {
        'e': 0,
        'syntype': 'Exp2Syn',
        'tau1': 10.,
        'tau2': 30.,
        'weight': 0.005,
        'record_current': False,
    }
    # Inhibitory synapse parameters
    syn_params_GABA_A = {
        'e': -80,
        'syntype': 'Exp2Syn',
        'tau1': 1.,
        'tau2': 12.,
        'weight': 0.005,
        'record_current': False
    }
    # where to insert, how many, and which input statistics
    syn_AMPA_args = {
        'section': 'allsec',
        'n': 100,
        'netstimParameters': {
            'number': 1000,
            'start': 0,
            'noise': 1,
            'interval': 20,
        }
    }
    syn_NMDA_args = {
        'section': ['dend', 'apic'],
        'n': 15,
        'netstimParameters': {
            'number': 1000,
            'start': 0,
            'noise': 1,
            'interval': 90,
        }
    }
    syn_GABA_A_args = {
        'section': 'dend',
        'n': 100,
        'netstimParameters': {
            'number': 1000,
            'start': 0,
            'noise': 1,
            'interval': 20,
        }
    }

    insert_synapses(cell, syn_params_AMPA, **syn_AMPA_args)
    insert_synapses(cell, syn_params_NMDA, **syn_NMDA_args)
    insert_synapses(cell, syn_params_GABA_A, **syn_GABA_A_args)


def realistic_stimuli_hay():

    cell_name = "hay"
    sampling_rate = 30000  # Hz
    dt = 1 / sampling_rate * 1000 / 4  # ???

    print("dt: ", dt)
    #dt = 2**-7
    # target numsteps: 82 * 4 = 328
    tstop = 2200
    cutoff = 200

    cell = return_hay_cell(tstop, dt)
    insert_distributed_synaptic_input(cell, weight_scale=1)
    cell.simulate(rec_vmem=True, rec_imem=True)
    t0 = np.argmin(np.abs(cell.tvec - cutoff))
    cell.tvec = cell.tvec[t0:] - cell.tvec[t0]
    cell.imem = cell.imem[:, t0:]
    cell.vmem = cell.vmem[:, t0:]
    cell.somav = cell.somav[t0:]

    plot_spikes(cell, cell_name)
    plot_single_cell_detectable_volume(cell, "hay", join("..", "model_scan"), fig_suptitle="Hay L5 PC")

def realistic_stimuli_hallermann():

    cell_name = "hallermann"
    # dt = 2**-7
    cutoff = 200
    tstop = 2200
    sampling_rate = 30000  # Hz
    dt = 1 / sampling_rate * 1000 / 4  # ???


    cell = return_hallermann_cell(tstop, dt)
    insert_distributed_synaptic_input(cell, weight_scale=0.1)
    cell.simulate(rec_vmem=True, rec_imem=True)
    t0 = np.argmin(np.abs(cell.tvec - cutoff))
    cell.tvec = cell.tvec[t0:] - cell.tvec[t0]
    cell.imem = cell.imem[:, t0:]
    cell.vmem = cell.vmem[:, t0:]
    cell.somav = cell.somav[t0:]

    plot_spikes(cell, cell_name)
    plot_single_cell_detectable_volume(cell, "hallermann", join("..", "model_scan"), fig_suptitle="Hallerman L5 PC")


def realistic_stimuli_BBP():

    sampling_rate = 30000  # Hz
    dt = 1 / sampling_rate * 1000 / 4  # ???
    # dt = 2**-7
    tstop = 200
    cutoff = 50
    weight_scale = 1
    cell_names = [f for f in os.listdir(bbp_folder) if f.startswith("L")]
    print(cell_names)
    if len(cell_names) == 0:
        raise RuntimeError("No cell models in folder!")
    for cell_name in cell_names:
        # if os.path.isfile(os.path.join(imem_eap_folder,
        #                                "imem_ufilt_%s.npy" % cell_name)):
        #     sim_success = True
        #     print("skipping ", cell_name)
        # else:
        sim_success = False
        weight_scale_ = weight_scale
        if "PC" in cell_name:
            weight_scale_ *= 1.5
        #while not sim_success:
        counter = 0
        while (not sim_success) and (counter < 20):
            pid = os.fork()
            if pid == 0:
                cell = return_BBP_neuron(cell_name, tstop, dt)
                # insert_distributed_synaptic_input(cell, weight_scale_)
                synapse, cell = insert_current_stimuli(cell, -0.15 * weight_scale_)
                cell.simulate(rec_vmem=True, rec_imem=True)

                t0 = np.argmin(np.abs(cell.tvec - cutoff))
                cell.tvec = cell.tvec[t0:] - cell.tvec[t0]
                cell.imem = cell.imem[:, t0:]
                cell.vmem = cell.vmem[:, t0:]
                cell.somav = cell.somav[t0:]
                plot_spikes(cell, cell_name)
                try:
                    plot_single_cell_detectable_volume(cell, "bbp_%s" % cell_name, join("..", "model_scan", "BBP"),
                                                   fig_suptitle=cell_name)
                except:
                    pass
                os._exit(0)
            else:
                os.waitpid(pid, 0)
                # plt.pause(0.1)
                if os.path.isfile(os.path.join(imem_eap_folder,
                                               "imem_ufilt_%s.npy" % cell_name)):
                    sim_success = True
                else:
                    weight_scale_ *= 1.5
                    counter += 1


def realistic_stimuli_allen():

    sampling_rate = 30000  # Hz
    dt = 1 / sampling_rate * 1000 / 4
    # dt = 2**-7
    tstop = 200
    cutoff = 50
    weight_scale = 1
    cell_names = [f[-9:] for f in os.listdir(allen_folder) if f.startswith("neuronal_model") and
                  os.path.isdir(join(allen_folder, f))]
    # cell_names = [f[15:-4] for f in os.listdir(allen_folder) if f.startswith("neuronal_model") and
    #               f.endswith(".zip")]

    print(cell_names)
    # sys.exit()
    if len(cell_names) == 0:
        raise RuntimeError("No cell models in folder!")
    for cell_name in cell_names:
        if os.path.isfile(os.path.join(imem_eap_folder,
                                       "imem_ufilt_%s.npy" % cell_name)):
            sim_success = True
            print("skipping ", cell_name)
        else:
            sim_success = False
        weight_scale_ = weight_scale
        counter = 0
        while (not sim_success) and (counter < 20):

            pid = os.fork()
            if pid == 0:
                model_folder = join(allen_folder, "neuronal_model_%s" % cell_name)
                if not os.path.isdir(model_folder):
                    download_allen_model(cell_name)
                cell = return_allen_cell_model(model_folder, dt, tstop)
                #insert_distributed_synaptic_input(cell, weight_scale_)
                synapse, cell = insert_current_stimuli(cell, -0.15 * weight_scale_)
                cell.simulate(rec_vmem=True, rec_imem=True)

                t0 = np.argmin(np.abs(cell.tvec - cutoff))
                cell.tvec = cell.tvec[t0:] - cell.tvec[t0]
                cell.imem = cell.imem[:, t0:]
                cell.vmem = cell.vmem[:, t0:]
                cell.somav = cell.somav[t0:]
                plot_spikes(cell, cell_name)
                os._exit(0)
            else:
                os.waitpid(pid, 0)
                # plt.pause(0.1)
                if os.path.isfile(os.path.join(imem_eap_folder,
                                               "imem_ufilt_%s.npy" % cell_name)):
                    sim_success = True
                else:
                    weight_scale_ *= 1.5
                    counter += 1


def control_sim_allen_cells():

    fig_folder = 'allen_control_sim'
    os.makedirs(fig_folder, exist_ok=True)
    dt = 2**-7
    tstop = 200
    cutoff = 0
    weight_scale = 1
    cell_names = [f[-9:] for f in os.listdir(allen_folder) if f.startswith("neuronal_model") and
                  os.path.isdir(join(allen_folder, f))]
    print(cell_names)
    if len(cell_names) == 0:
        raise RuntimeError("No cell models in folder!")
    for cell_name in cell_names:
        sim_success = False
        weight_scale_ = weight_scale
        while not sim_success:
            pid = os.fork()
            if pid == 0:
                model_folder = join(allen_folder, "neuronal_model_%s" % cell_name)
                if not os.path.isdir(model_folder):
                    download_allen_model(cell_name)
                cell = return_allen_cell_model(model_folder, dt, tstop)
                #insert_distributed_synaptic_input(cell, weight_scale_)
                synapse, cell = insert_current_stimuli(cell, -0.15 * weight_scale_)
                cell.simulate(rec_vmem=True, rec_imem=True)

                spike_time_idxs = return_spike_time_idxs(cell.somav)
                if len(spike_time_idxs) == 0:
                    print(cell_name, " not spiking!")
                    sim_success = False
                else:
                    sim_success = True

                plt.close("all")
                fig = plt.figure(figsize=[16, 9])
                ax_m = fig.add_axes([0.0, 0., 0.15, 0.97], aspect=1,
                                    frameon=False, xticks=[], yticks=[])
                ax_v = fig.add_axes([0.25, 0.15, 0.7, 0.7])
                ax_m.plot(cell.x.T, cell.z.T, c='k')
                [ax_v.plot(cell.tvec, cell.vmem[idx, :], 'gray', lw=0.5) for idx in range(cell.totnsegs)]
                ax_v.plot(cell.tvec, cell.vmem[0, :], 'k', lw=1)
                print(cell.somaidx)
                fig.savefig(join(fig_folder, "allen_test_%s_%s.png" % (cell_name, sim_success)))
                #plot_spikes(cell, cell_name)
                cell.__del__()
                os._exit(0)
            else:
                os.waitpid(pid, 0)
                # plt.pause(0.1)
                if os.path.isfile(join(fig_folder, "allen_test_%s_%s.png" % (cell_name, True))):
                    sim_success = True
                else:
                    weight_scale_ *= 1.5


def simulate_passing_axon():

    dt = 2**-7
    tstop = 15
    cutoff = 0
    axon_type = "myelinated"
    cell_name = "%s_axon" % axon_type
    from ECSbook_simcode import hallermann_axon_model as ha_ax
    if axon_type == "unmyelinated":
        cell = ha_ax.return_constructed_unmyelinated_axon(dt, tstop, 0, 1, axon_diameter=None)
    elif axon_type == "myelinated":
        cell = ha_ax.return_constructed_myelinated_axon(dt, tstop, 0, 1, axon_diameter=None)
    else:
        raise RuntimeError("axon_type not recognized")

    insert_current_stimuli(cell, -0.03)
    cell.set_pos(z=-200)
    cell.simulate(rec_vmem=True, rec_imem=True)

    t0 = np.argmin(np.abs(cell.tvec - cutoff))
    cell.tvec = cell.tvec[t0:] - cell.tvec[t0]
    cell.imem = cell.imem[:, t0:]
    cell.vmem = cell.vmem[:, t0:]
    cell.somav = cell.vmem[0, t0:]
    plot_spikes(cell, cell_name)


def plot_spikes(cell, cell_name):
    num_elecs = 1
    elec_params = {
        'sigma': sigma,  # Saline bath conductivity
        'x': np.array([20]),  # electrode requires 1d vector of positions
        'y': np.zeros(num_elecs),
        'z': np.zeros(num_elecs),
        "method": "root_as_point",
    }
    filt_dict_high_pass = {'highpass_freq': 300,
                           'lowpass_freq': None,
                           'order': 1,
                           'filter_function': 'filtfilt',
                           'fs': 1 / (cell.dt / 1000),
                           'axis': -1
                           }

    # high-pass filter settings
    Fs = 1000 / cell.dt
    N = 1  # filter order
    rp = 0.1  # ripple in passband (dB)
    rs = 40.  # minimum attenuation required in the stop band (dB)
    fc = 300.  # critical frequency (Hz)

    # filter coefficients on 'sos' format
    sos_ellip = ss.ellip(N=N, rp=rp, rs=rs, Wn=fc, btype='hp', fs=Fs, output='sos')

    electrode = LFPy.RecExtElectrode(cell, **elec_params)
    imem_filt = elephant.signal_processing.butter(cell.imem, **filt_dict_high_pass)
    imem_filt2 = ss.sosfiltfilt(sos_ellip, cell.imem)
    v_e_ufilt = electrode.get_transformation_matrix() @ cell.imem * 1e3
    v_e_filt = elephant.signal_processing.butter(v_e_ufilt, **filt_dict_high_pass)

    spike_time_idxs = return_spike_time_idxs(cell.somav)
    # print(cell.tvec[spike_time_idxs])
    window_len = 2.7
    window_t0 = -int(1.27 / cell.dt)
    window_t1 = int(window_len / cell.dt + 1) + window_t0
    # print(window_t0 * cell.dt, window_t1 * cell.dt)

    if len(spike_time_idxs) == 0:
        print(cell_name, " not spiking!")
        is_spiking = False
        spike_windows = np.array([[0, len(cell.tvec) - 1]])
        #return None
    else:
        is_spiking = True
        spike_windows = np.array([spike_time_idxs + window_t0,
                                  spike_time_idxs + window_t1]).T

    # print(spike_windows)
    eaps_filt = []
    eaps_ufilt = []
    imems_filt = []
    imems_filt2 = []
    imems_ufilt = []
    for s_wind in spike_windows:
        if s_wind[0] >= 0 and s_wind[1] < len(cell.tvec):
            eaps_filt.append(v_e_filt[:, s_wind[0]:s_wind[1]])
            eaps_ufilt.append(v_e_ufilt[:, s_wind[0]:s_wind[1]])
            imems_filt.append(imem_filt[:, s_wind[0]:s_wind[1]])
            imems_filt2.append(imem_filt2[:, s_wind[0]:s_wind[1]])
            imems_ufilt.append(cell.imem[:, s_wind[0]:s_wind[1]])

    imem_filt_mean = np.mean(imems_filt, axis=0)
    imem_filt2_mean = np.mean(imems_filt2, axis=0)
    imem_ufilt_mean = np.mean(imems_ufilt, axis=0)
    if is_spiking:
        np.save(os.path.join(imem_eap_folder, "imem_ufilt_%s.npy" % cell_name), imem_ufilt_mean)
        np.save(os.path.join(imem_eap_folder, "imem_filt_%s.npy" % cell_name), imem_filt_mean)
        np.save(os.path.join(imem_eap_folder, "imem_filt2_%s.npy" % cell_name), imem_filt2_mean)

    v_e_prefilt = electrode.get_transformation_matrix() @ imem_filt_mean * 1e3
    v_e_prefilt2 = electrode.get_transformation_matrix() @ imem_filt2_mean * 1e3

    # print("neg Ve peak time: ", cell.tvec[np.argmin(v_e_prefilt2[0])])

    #print(spike_windows)
    eaps_filt = np.array(eaps_filt)
    eaps_ufilt = np.array(eaps_ufilt)
    plt.close("all")
    fig = plt.figure(figsize=[16, 9])
    ax_m = fig.add_axes([0.0, 0., 0.15, 0.97], aspect=1,
                        frameon=False, xticks=[], yticks=[])
    ax_v = fig.add_axes([0.25, 0.6, 0.2, 0.3])
    ax_v_e = fig.add_axes([0.25, 0.1, 0.2, 0.3])
    ax_eap = fig.add_axes([0.5, 0.1, 0.45, 0.8])
    ax_m.plot(cell.x.T, cell.z.T, c='k')
    ax_v.plot(cell.tvec, cell.somav, 'k')
    ax_m.plot(electrode.x, electrode.z, 'D', c='orange')

    [ax_v.axvline(cell.tvec[t_idx]) for t_idx in spike_time_idxs]
    [ax_v_e.axvline(cell.tvec[t_idx]) for t_idx in spike_time_idxs]

    for elec in range(num_elecs):
        ax_v_e.plot(cell.tvec, v_e_ufilt[elec], 'k', lw=2)
        ax_v_e.plot(cell.tvec, v_e_filt[elec], 'r', lw=1.5)

    t_eap = cell.tvec[:eaps_filt.shape[-1]]
    for eap_idx in range(len(eaps_filt)):
        ax_eap.plot(t_eap, eaps_filt[eap_idx, 0] - eaps_filt[eap_idx, 0, 0], c='pink', lw=0.5)
        ax_eap.plot(t_eap, eaps_ufilt[eap_idx, 0] - eaps_ufilt[eap_idx, 0, 0], c='gray', lw=0.5)
    mean_eap_filt = np.mean(eaps_filt, axis=0)
    mean_eap_ufilt = np.mean(eaps_ufilt, axis=0)
    l1, = ax_eap.plot(t_eap, mean_eap_filt[0] - mean_eap_filt[0, 0], c='r', lw=2)
    l2, = ax_eap.plot(t_eap, mean_eap_ufilt[0] - mean_eap_ufilt[0, 0], c='k', lw=2)
    ax_eap.plot(t_eap, v_e_prefilt[0] - v_e_prefilt[0, 0], 'b--', lw=1)
    ax_eap.plot(t_eap, v_e_prefilt2[0] - v_e_prefilt2[0, 0], 'g--', lw=1)
    ax_eap.legend([l1, l2], ["hp-filtered", "unfiltered"], frameon=False)
    fig_folder = os.path.join("..", "sim_control_figs")
    os.makedirs(fig_folder, exist_ok=True)
    plt.savefig(join(fig_folder, "sim_EAP_%s_one_pole_%s.png" % (cell_name, is_spiking)))
    # plt.show()

if __name__ == '__main__':

    # realistic_stimuli_hay()
    # realistic_stimuli_hallermann()
    # realistic_stimuli_BBP()
    # realistic_stimuli_allen()

    # run_chosen_allen_models()
    # control_sim_allen_cells()
    # recreate_allen_data_axon()
    # simulate_passing_axon()
    # inspect_cells()
    # analyze_eap_amplitudes()

    # recreate_allen_data_hay()
    # recreate_allen_data_hallermann()
    recreate_allen_data_BBP()
    # recreate_allen_data()

