"""Retrieve Allen Cell Types biophysical models by cell type.

Queries the Allen Cell Types database (https://celltypes.brain-map.org) through
its public RMA API and downloads every *biophysical* neuronal model matching a
set of filters (species / cortical layer / dendrite type) into

    cell_models/allen_models/neuronal_model_<id>/

which is exactly the layout that ``main.return_allen_cell_model`` expects and
that ``main.control_sim_allen_cells`` scans. So the intended workflow is:

    python retrieve_allen_cells.py --species mouse --layer 2/3 --dendrite spiny
    # ...then, in main.py, run:
    control_sim_allen_cells()

Only depends on the standard library (urllib/zipfile), so it works even when
the AllenSDK is not installed (it does not install cleanly on recent Pythons).

Notes on the two-step lookup
----------------------------
The API distinguishes a *specimen* (a recorded cell) from a *neuronal model*
fit to that cell. The ``neuronal_model/download`` endpoint takes a *model* id,
not a specimen id, and a single specimen can have several models (perisomatic,
all-active, and various GLIF point-neuron models). Only the two biophysical
templates below can be simulated by ``return_allen_cell_model`` -- the GLIF
models are point neurons with no morphology and are skipped.
"""

import os
import ssl
import json
import shutil
import zipfile
import argparse
import subprocess
import urllib.parse
import urllib.request
from os.path import join, dirname, abspath, isdir, isfile

# Same paths main.py uses, so downloads land where control_sim_allen_cells looks.
_this_dir = dirname(abspath(__file__))
cell_models_folder = join(_this_dir, "cell_models")
allen_folder = join(cell_models_folder, "allen_models")

API_BASE = "https://api.brain-map.org/api/v2/data/query.json"
DOWNLOAD_URL = "https://api.brain-map.org/neuronal_model/download/%s"

# Templates that return_allen_cell_model can actually build (everything else,
# e.g. the "Leaky Integrate and Fire" GLIF families, is a point neuron).
BIOPHYSICAL_TEMPLATES = (
    "Biophysical - all active",
    "Biophysical - perisomatic",
)

# Convenience aliases -> exact strings stored in the database.
SPECIES_ALIASES = {
    "mouse": "Mus musculus",
    "mus musculus": "Mus musculus",
    "human": "Homo Sapiens",
    "homo sapiens": "Homo Sapiens",
}

# RMA query strings contain these structural characters; keep them un-escaped
# (only spaces and the like get percent-encoded).
_SAFE = "::,[]$'()."

_ssl_ctx = ssl._create_unverified_context()


def _rma(criteria):
    """Run one RMA query, return the list of result rows, raise on API error."""
    url = API_BASE + "?criteria=" + urllib.parse.quote(criteria, safe=_SAFE)
    with urllib.request.urlopen(url, context=_ssl_ctx, timeout=60) as resp:
        result = json.load(resp)
    if not result.get("success", False):
        raise RuntimeError("Allen API query failed: %s\n  %s"
                           % (result.get("msg"), url))
    return result["msg"]


def find_specimens(species="Mus musculus", layer=None, dendrite_type=None):
    """Return specimen rows that have at least one biophysical model.

    ``layer`` (e.g. "2/3", "4", "5") and ``dendrite_type`` (e.g. "spiny",
    "aspiny", "sparsely spiny") are optional; omit them to widen the search.
    """
    species = SPECIES_ALIASES.get(species.strip().lower(), species)

    conditions = ["[donor__species$eq'%s']" % species]
    if layer:
        conditions.append("[structure__layer$eq'%s']" % layer)
    if dendrite_type:
        conditions.append("[tag__dendrite_type$eq'%s']" % dendrite_type)

    # A specimen has a biophysical model iff one of these flags is set. RMA
    # ANDs bracketed conditions, so we query each flag separately and merge --
    # this is the union "perisomatic OR all-active".
    specimens = {}
    for flag in ("m__biophys_perisomatic", "m__biophys_all_active"):
        criteria = ("model::ApiCellTypesSpecimenDetail,rma::criteria,"
                    + "".join(conditions) + "[%s$eq1]," % flag
                    + "rma::options[num_rows$eq'all']")
        for row in _rma(criteria):
            specimens[row["specimen__id"]] = row
    return list(specimens.values())


def find_biophysical_model_ids(specimen_id, model_types=BIOPHYSICAL_TEMPLATES):
    """Return [(model_id, template_name), ...] for one specimen."""
    criteria = ("model::NeuronalModel,rma::criteria,[specimen_id$eq%s],"
                "rma::include,neuronal_model_template" % specimen_id)
    models = []
    for nm in _rma(criteria):
        template = (nm.get("neuronal_model_template") or {}).get("name", "")
        if template in model_types:
            models.append((nm["id"], template))
    return models


def download_allen_model(model_id, compile_mod=True):
    """Download and unpack one neuronal model into allen_folder.

    Standalone re-implementation of main.download_allen_model that uses urllib
    instead of shelling out to ``wget`` (which is not always installed) and
    tolerates a missing ``nrnivmodl`` -- the mechanisms are compiled lazily by
    return_allen_cell_model on first use anyway.
    """
    model_id = str(model_id)
    os.makedirs(allen_folder, exist_ok=True)
    model_folder = join(allen_folder, "neuronal_model_%s" % model_id)
    if isdir(model_folder):
        print("  already present: neuronal_model_%s" % model_id)
        return model_folder

    print("  downloading neuronal_model_%s ..." % model_id)
    tmp_zip = join(allen_folder, "neuronal_model_%s.zip" % model_id)
    with urllib.request.urlopen(DOWNLOAD_URL % model_id,
                                context=_ssl_ctx, timeout=120) as resp:
        data = resp.read()
    with open(tmp_zip, "wb") as fh:
        fh.write(data)
    try:
        with zipfile.ZipFile(tmp_zip) as zf:
            zf.extractall(model_folder)
    finally:
        os.remove(tmp_zip)

    if compile_mod:
        _compile_mechanisms(model_folder)
    return model_folder


def _compile_mechanisms(model_folder):
    mod_folder = join(model_folder, "modfiles")
    if not isdir(mod_folder) or isdir(join(mod_folder, "x86_64")):
        return
    nrnivmodl = shutil.which("nrnivmodl")
    if nrnivmodl is None:
        print("    (nrnivmodl not found -- will compile on first simulation)")
        return
    print("    compiling mechanisms ...")
    subprocess.run([nrnivmodl], cwd=mod_folder, check=True)


def _ensure_remove_axon_hoc():
    """control_sim_allen_cells builds cells with custom_code pointing at
    allen_folder/remove_axon.hoc, but the file ships in cell_models/. Copy it
    into place so the simulation step does not fail on a missing hoc file."""
    src = join(cell_models_folder, "remove_axon.hoc")
    dst = join(allen_folder, "remove_axon.hoc")
    if isfile(src) and not isfile(dst):
        os.makedirs(allen_folder, exist_ok=True)
        shutil.copy(src, dst)
        print("Copied remove_axon.hoc into allen_models/")


def retrieve_allen_cells(species="Mus musculus", layer=None, dendrite_type=None,
                         model_types=BIOPHYSICAL_TEMPLATES, max_cells=None,
                         download=True, compile_mod=True):
    """Find and download all biophysical Allen models of a given cell type.

    Returns the list of neuronal-model ids that were found (and downloaded, if
    ``download`` is True). After this you can run ``control_sim_allen_cells()``.
    """
    print("Searching Allen Cell Types for: species=%r layer=%r dendrite=%r"
          % (species, layer, dendrite_type))
    specimens = find_specimens(species, layer, dendrite_type)
    print("Found %d specimen(s) with a biophysical model." % len(specimens))

    found = []  # (specimen_id, model_id, template)
    for spec in specimens:
        sid = spec["specimen__id"]
        for model_id, template in find_biophysical_model_ids(sid, model_types):
            found.append((sid, model_id, template))

    if max_cells is not None:
        found = found[:max_cells]

    print("Matched %d biophysical model(s):" % len(found))
    for sid, model_id, template in found:
        print("  specimen %s -> model %s [%s]" % (sid, model_id, template))

    model_ids = [model_id for _, model_id, _ in found]
    if download and model_ids:
        _ensure_remove_axon_hoc()
        print("Downloading into %s" % allen_folder)
        for model_id in model_ids:
            try:
                download_allen_model(model_id, compile_mod=compile_mod)
            except Exception as exc:  # keep going on a single bad download
                print("  FAILED model %s: %s" % (model_id, exc))
    elif download:
        print("Nothing to download.")

    return model_ids


def _parse_args():
    p = argparse.ArgumentParser(
        description="Download Allen Cell Types biophysical models by cell type.")
    p.add_argument("--species", default="mouse",
                   help="e.g. 'mouse' / 'Mus musculus' / 'human' (default: mouse)")
    p.add_argument("--layer", default=None,
                   help="cortical layer, e.g. '2/3', '4', '5', '6a' (default: any)")
    p.add_argument("--dendrite", dest="dendrite_type", default=None,
                   help="'spiny', 'aspiny', or 'sparsely spiny' (default: any)")
    p.add_argument("--model-types", nargs="+", default=list(BIOPHYSICAL_TEMPLATES),
                   choices=list(BIOPHYSICAL_TEMPLATES),
                   help="which biophysical templates to keep (default: both)")
    p.add_argument("--max-cells", type=int, default=None,
                   help="limit number of models downloaded")
    p.add_argument("--list-only", action="store_true",
                   help="only list matches, do not download")
    p.add_argument("--no-compile", action="store_true",
                   help="skip nrnivmodl compilation of .mod files")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    retrieve_allen_cells(
        species=args.species,
        layer=args.layer,
        dendrite_type=args.dendrite_type,
        model_types=tuple(args.model_types),
        max_cells=args.max_cells,
        download=not args.list_only,
        compile_mod=not args.no_compile,
    )
