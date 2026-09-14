"""Retrieve Blue Brain Project (NMC portal) cell models by cell type.

Downloads the neocortical microcircuit "hoc_combos" models from the EPFL NMC
portal and unpacks the selected cells into

    cell_models/bbp_models/<cell_name>/

which is exactly the layout that ``main.return_BBP_neuron`` expects (each folder
holds ``template.hoc``, ``biophysics.hoc``, ``morphology.hoc``, ``constants.hoc``,
``synapses/``, ``morphology/`` and ``mechanisms/``). All unique ``.mod`` files are
then gathered into ``cell_models/bbp_mod/`` and compiled once with ``nrnivmodl``,
matching ``main.compile_bbp_mechanisms``. So the intended workflow is:

    python retrieve_bbp_cells.py --layer L5 --mtype TTPC2
    # ...then, in main.py, run e.g.:
    realistic_stimuli_BBP()

Why this exists
---------------
``main.download_BBP_model`` fetches one cell at a time from the per-cell
``downloads-zip/<cell_name>.zip`` endpoint, which is no longer reliable. This
script instead uses the bulk archive that the portal still serves:

    https://bbp.epfl.ch/.../Download/hoc_combos_syn.1_0_10.allzips.tar   (~786 MB)

The tar contains one ``<cell_name>.zip`` per model (1035 in total). We stream the
tar and only unpack the zips matching the requested filters, so a subset can be
retrieved without keeping the whole archive on disk. Only depends on the standard
library (urllib / tarfile / zipfile).

Cell naming convention
----------------------
``L5_TTPC2_cADpyr232_1`` -> layer ``L5`` / m-type ``TTPC2`` / e-type ``cADpyr232``
/ clone ``1``. The filters below match on these fields.
"""

import io
import os
import re
import sys
import ssl
import glob
import shutil
import tarfile
import zipfile
import argparse
import subprocess
import urllib.request
from os.path import join, dirname, abspath, isdir, isfile, getsize

# Same paths main.py uses, so downloads land where return_BBP_neuron looks.
_this_dir = dirname(abspath(__file__))
cell_models_folder = join(_this_dir, "cell_models")
bbp_folder = join(cell_models_folder, "bbp_models")
bbp_mod_folder = join(cell_models_folder, "bbp_mod")

TAR_URL = ("https://bbp.epfl.ch/nmc-portal/assets/documents/static/"
           "Download/hoc_combos_syn.1_0_10.allzips.tar")

_ssl_ctx = ssl._create_unverified_context()


def _parse_cell_name(cell_name):
    """``L5_TTPC2_cADpyr232_1`` -> ('L5', 'TTPC2', 'cADpyr232', '1').

    Returns ('', '', '', '') for anything that is not a 4-field BBP name.
    """
    parts = cell_name.split("_")
    if len(parts) != 4:
        return ("", "", "", "")
    return tuple(parts)


def _matches(cell_name, layer, mtype, etype, name):
    """Decide whether a cell passes the (all-optional) filters."""
    c_layer, c_mtype, c_etype, _clone = _parse_cell_name(cell_name)
    if layer and c_layer.lower() != layer.lower():
        return False
    if mtype and c_mtype.lower() != mtype.lower():
        return False
    # e-type carries a trailing number (cADpyr232); match on the prefix so
    # "cADpyr" selects the whole family.
    if etype and not c_etype.lower().startswith(etype.lower()):
        return False
    if name and name.lower() not in cell_name.lower():
        return False
    return True


def _member_cell_name(member):
    """'hoc_combos_syn.1_0_10.allzips/L5_TTPC2_cADpyr232_1.zip' -> cell name,
    or None for non-zip members (e.g. the top-level directory entry)."""
    base = os.path.basename(member.name)
    if not base.endswith(".zip"):
        return None
    return base[:-len(".zip")]


def _download_tar(tar_path):
    """Download the bulk tar to ``tar_path`` with HTTP-range resume support."""
    os.makedirs(dirname(tar_path) or ".", exist_ok=True)
    have = getsize(tar_path) if isfile(tar_path) else 0

    req = urllib.request.Request(TAR_URL)
    if have:
        req.add_header("Range", "bytes=%d-" % have)
        print("Resuming tar download at %d bytes ..." % have)
    else:
        print("Downloading tar (%s) ..." % TAR_URL)

    with urllib.request.urlopen(req, context=_ssl_ctx, timeout=120) as resp:
        # 200 => server ignored the Range header, restart from scratch.
        mode = "ab" if (have and resp.status == 206) else "wb"
        if mode == "wb":
            have = 0
        total = have + int(resp.headers.get("Content-Length", 0) or 0)
        chunk = 1 << 20  # 1 MiB
        with open(tar_path, mode) as fh:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                fh.write(buf)
                have += len(buf)
                if total:
                    print("\r  %6.1f / %6.1f MB (%4.1f%%)"
                          % (have / 1e6, total / 1e6, 100 * have / total),
                          end="", flush=True)
        print()
    return tar_path


def _open_tar(tar_path=None, force_download=False):
    """Return (tarfile, close_extra) ready to iterate over.

    * ``tar_path`` given and present (and not ``force_download``): open it
      locally with random access.
    * ``tar_path`` given but missing: download the full tar there first
      (resumable), then open it -- handy when re-running with new filters.
    * no ``tar_path``: stream straight from the URL, so nothing large is
      written to disk.
    """
    if tar_path and isfile(tar_path) and not force_download:
        print("Using local tar: %s" % tar_path)
        return tarfile.open(tar_path, "r:"), None
    if tar_path:
        _download_tar(tar_path)
        return tarfile.open(tar_path, "r:"), None
    print("Streaming tar from %s" % TAR_URL)
    resp = urllib.request.urlopen(TAR_URL, context=_ssl_ctx, timeout=120)
    return tarfile.open(fileobj=resp, mode="r|"), resp


# BBP's stochastic-synapse mechanisms declare nrn_random_pick/nrn_random_arg in a
# VERBATIM block and call scop_random(1). Both break under NEURON >= 8.2/9, which
# now declares those symbols itself (with a different signature) and made
# scop_random take no argument. NEURON ships the NRN_VERSION_GTEQ_8_2_0 macro
# expressly for guarding such VERBATIM adaptations, so this port stays valid on
# older NEURON too.
_SYNAPSE_MODS = ("ProbAMPANMDA_EMS.mod", "ProbGABAAB_EMS.mod")

_RANDOM_PROTO = ("double nrn_random_pick(void* r);\n"
                 "void* nrn_random_arg(int argpos);\n")
_RANDOM_PROTO_GUARDED = ("#ifndef NRN_VERSION_GTEQ_8_2_0\n"
                         + _RANDOM_PROTO + "#endif\n")


def port_synapse_mod_for_neuron9(mod_path):
    """Make one BBP synapse .mod file compile under NEURON >= 8.2/9, in place.

    Idempotent: returns True if the file was changed, False if it was already
    ported (or is not one of the affected files)."""
    if not isfile(mod_path):
        return False
    src = open(mod_path).read()
    out = src
    # Guard the manual RNG prototypes (skip if already guarded).
    if "NRN_VERSION_GTEQ_8_2_0" not in out and _RANDOM_PROTO in out:
        out = out.replace(_RANDOM_PROTO, _RANDOM_PROTO_GUARDED)
    # scop_random(1) -> scop_random() (the arg was dropped in modern NEURON).
    out = re.sub(r"scop_random\s*\(\s*1\s*\)", "scop_random()", out)
    if out != src:
        open(mod_path, "w").write(out)
        return True
    return False


def _extract_cell(tar, member, overwrite=False):
    """Unpack one <cell_name>.zip member into bbp_folder. Returns the cell name."""
    cell_name = _member_cell_name(member)
    cell_folder = join(bbp_folder, cell_name)
    if isdir(cell_folder) and not overwrite:
        print("  already present: %s" % cell_name)
        return cell_name
    fobj = tar.extractfile(member)
    if fobj is None:
        return None
    data = fobj.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(bbp_folder)
    # Port the synapse mechanisms so nrnivmodl succeeds under NEURON >= 9.
    for mod_name in _SYNAPSE_MODS:
        port_synapse_mod_for_neuron9(join(cell_folder, "mechanisms", mod_name))
    print("  extracted: %s" % cell_name)
    return cell_name


def _find_nrnivmodl():
    """Locate a working nrnivmodl. Prefer the one shipped next to the running
    Python interpreter (a pip-installed NEURON in this venv) over whatever is on
    PATH, which may be a stale global install with a broken shebang."""
    local = join(dirname(sys.executable), "nrnivmodl")
    if isfile(local) and os.access(local, os.X_OK):
        return local
    return shutil.which("nrnivmodl")


def compile_bbp_mechanisms(cell_names):
    """Gather every unique .mod from the given cells into bbp_mod_folder and
    compile once with nrnivmodl (standalone version of the same-named helper in
    main.py). Skips compilation gracefully if nrnivmodl is unavailable."""
    os.makedirs(bbp_mod_folder, exist_ok=True)
    copied = 0
    for cell_name in cell_names:
        mech_dir = join(bbp_folder, cell_name, "mechanisms")
        for nmodl in glob.glob(join(mech_dir, "*.mod")):
            dst = join(bbp_mod_folder, os.path.basename(nmodl))
            if not isfile(dst):
                shutil.copy(nmodl, dst)
                copied += 1

    # Make sure the aggregated synapse mechanisms are NEURON >= 9 compatible,
    # even if they were copied from an already-present (un-ported) cell folder.
    for mod_name in _SYNAPSE_MODS:
        port_synapse_mod_for_neuron9(join(bbp_mod_folder, mod_name))

    compiled = isdir(join(bbp_mod_folder, "x86_64"))
    if compiled and copied == 0:
        print("Mechanisms already compiled (%s)." % bbp_mod_folder)
        return

    nrnivmodl = _find_nrnivmodl()
    if nrnivmodl is None:
        print("(nrnivmodl not found -- .mod files copied but not compiled; "
              "they will be compiled on first simulation.)")
        return
    print("Compiling %d mechanism(s) in %s ..."
          % (len(glob.glob(join(bbp_mod_folder, "*.mod"))), bbp_mod_folder))
    # Don't lose a successful download to a compile failure: warn and let the
    # caller decide, rather than raising. (The BBP synapse mechanisms are ported
    # for NEURON >= 9 above; any remaining failure is something else.)
    result = subprocess.run([nrnivmodl], cwd=bbp_mod_folder)
    if result.returncode != 0:
        print("WARNING: nrnivmodl failed (exit %d). Cells were downloaded, but "
              "mechanisms are not compiled." % result.returncode)


def retrieve_bbp_cells(layer=None, mtype=None, etype=None, name=None,
                       max_cells=None, tar_path=None, force_download=False,
                       download=True, compile_mod=True, overwrite=False):
    """Find and unpack all BBP cells matching the given filters.

    Returns the list of cell names that matched (and were extracted, if
    ``download`` is True). After this you can run e.g. ``realistic_stimuli_BBP()``.
    """
    print("Searching BBP models for: layer=%r mtype=%r etype=%r name=%r"
          % (layer, mtype, etype, name))
    os.makedirs(bbp_folder, exist_ok=True)

    tar, extra = _open_tar(tar_path, force_download)
    matched = []
    try:
        for member in tar:
            cell_name = _member_cell_name(member)
            if cell_name is None:
                continue
            if not _matches(cell_name, layer, mtype, etype, name):
                continue
            if max_cells is not None and len(matched) >= max_cells:
                break
            matched.append(cell_name)
            if download:
                _extract_cell(tar, member, overwrite=overwrite)
            else:
                print("  match: %s" % cell_name)
    finally:
        tar.close()
        if extra is not None:
            extra.close()

    print("Matched %d cell(s)." % len(matched))
    if download and matched and compile_mod:
        compile_bbp_mechanisms(matched)
    return matched


def _parse_args():
    p = argparse.ArgumentParser(
        description="Download Blue Brain Project (NMC portal) cell models by "
                    "cell type from the bulk hoc_combos archive.")
    p.add_argument("--layer", default=None,
                   help="cortical layer, e.g. 'L1'..'L6' (default: any)")
    p.add_argument("--mtype", default=None,
                   help="morphological type, e.g. 'TTPC2', 'MC', 'NGC' (default: any)")
    p.add_argument("--etype", default=None,
                   help="electrical type prefix, e.g. 'cADpyr', 'bAC' (default: any)")
    p.add_argument("--name", default=None,
                   help="substring match on the full cell name (default: any)")
    p.add_argument("--max-cells", type=int, default=None,
                   help="limit number of cells extracted")
    p.add_argument("--tar-path", default=None,
                   help="local path for the bulk tar; reused if present, "
                        "downloaded there if missing (default: stream, no cache)")
    p.add_argument("--force-download", action="store_true",
                   help="re-download the tar even if --tar-path exists")
    p.add_argument("--overwrite", action="store_true",
                   help="re-extract cells whose folder already exists")
    p.add_argument("--list-only", action="store_true",
                   help="only list matches, do not extract "
                        "(still streams the whole tar to read its index)")
    p.add_argument("--no-compile", action="store_true",
                   help="skip nrnivmodl compilation of .mod files")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    retrieve_bbp_cells(
        layer=args.layer,
        mtype=args.mtype,
        etype=args.etype,
        name=args.name,
        max_cells=args.max_cells,
        tar_path=args.tar_path,
        force_download=args.force_download,
        download=not args.list_only,
        compile_mod=not args.no_compile,
        overwrite=args.overwrite,
    )
