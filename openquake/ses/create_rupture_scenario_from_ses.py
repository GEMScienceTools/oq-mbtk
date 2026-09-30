# ------------------- The OpenQuake Model Building Toolkit --------------------
# Copyright (C) 2022 GEM Foundation
#           _______  _______        __   __  _______  _______  ___   _
#          |       ||       |      |  |_|  ||  _    ||       ||   | | |
#          |   _   ||   _   | ____ |       || |_|   ||_     _||   |_| |
#          |  | |  ||  | |  ||____||       ||       |  |   |  |      _|
#          |  |_|  ||  |_|  |      |       ||  _   |   |   |  |     |_
#          |       ||      |       | ||_|| || |_|   |  |   |  |    _  |
#          |_______||____||_|      |_|   |_||_______|  |___|  |___| |_|
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------
# vim: tabstop=4 shiftwidth=4 softtabstop=4
# coding: utf-8

"""
Extract rupture from SES

Selects a single rupture (by ID) from an OpenQuake SES/GMF datastore,
writes it out as a gridded-rupture NRML file, and builds the matching
GMC (ground-motion characterization) logic-tree NRML file for the
tectonic region type of the source that generated it.

Usage
-----
    python create_rupture_scenario_from_ses.py config.toml
    python create_rupture_scenario_from_ses.py config.toml --rupture-id 615649202388021
    python create_rupture_scenario_from_ses.py --help

Configuration File
------------------

Below we provide an example of the .toml file containing the configuration
settings:

```
# HDF5 datastore containing the SESs and GMFs
datastore = "/Users/mpagani/oqdata/calc_248.hdf5"

# ID of the rupture to select
rupture_id = 615649202388021

# Ratio used when adding between/within std devs to GMMs that don't
# natively provide them (add_between_within_stds)
with_between_within_ratio = 1.4

# Output files
output_rupture_xml = "rupture_model.xml"
output_gmclt_xml = "gmclt.xml"
```

"""

try:
    import tomllib  # Python >= 3.11
except ImportError:  # pragma: no cover
    import tomli as tomllib  # Python < 3.11

import numpy as np

from openquake.baselib import sap
from openquake.hazardlib import valid
from openquake.hazardlib.source.rupture import to_arrays
from openquake.commonlib.datastore import read as dstore_read
from openquake.hazardlib.gsim.mgmpe.modifiable_gmpe import ModifiableGMPE
from openquake.hazardlib.source.rupture import BaseRupture


# -----------------------------------------------------------------------
# Templates
# -----------------------------------------------------------------------
FMT_NRML = """<?xml version="1.0" encoding="utf-8"?>
<nrml xmlns:gml="http://www.opengis.net/gml"
   xmlns="http://openquake.org/xmlns/nrml/0.5">
{content}
</nrml>
"""

FMT_RUP = """   <griddedRupture>
      <magnitude>{mag:.2f}</magnitude>
      <rake>{rake:.2f}</rake>
      <hypocenter depth="{dep:.2f}" lat="{lat:.6f}" lon="{lon:.6f}"/>
      <griddedSurface surface_type="{stype}" rupture_type="{stype}">
         <gml:posList>
            {coos}
         </gml:posList>
      </griddedSurface>
   </griddedRupture>"""

FMT_BRANCH = """      <logicTreeBranch branchID="{bid}">
         <uncertaintyModel>
            {u_model}
         </uncertaintyModel>
         <uncertaintyWeight>{u_weight}</uncertaintyWeight>
      </logicTreeBranch>
"""

FMT_GMC_LT = """<logicTree logicTreeID="lt1">
   <logicTreeBranchSet uncertaintyType="gmpeModel"
                        branchSetID="bs1"
                        applyToTectonicRegionType="{trt_lab}">
{branches}   </logicTreeBranchSet>
</logicTree>"""

# Default names
DEFAULTS = {
    'output_rupture_xml': 'rupture_model.xml',
    'output_gmclt_xml': 'gmclt.xml',
}


def load_config(path):
    """
    Read the TOML configuration file and fill in missing keys with
    the defaults.
    """
    with open(path, 'rb') as fh:
        cfg = tomllib.load(fh) or {}
    out = dict(DEFAULTS)
    out.update(cfg)
    return out


def select_rupture(rups_data, rups_geom, find_rup_id):
    """
    :returns: (rup_idx, rup_meshes) for the rupture with id `find_rup_id`
    """
    rup_idx = np.where(rups_data['id'] == find_rup_id)[0]
    assert len(rup_idx) == 1, (
        f'expected exactly one rupture with id {find_rup_id}, '
        f'found {len(rup_idx)}')
    idx = rup_idx[0]
    print(f'Rupture index: {idx}')

    print('\nRupture Information')
    print('-------------------')
    for i, name in enumerate(rups_data.dtype.names):
        print(f"{name:20s}: {rups_data[idx][i]}")

    rgeom = rups_geom[rups_data[idx]['geom_id']]
    rup_meshes = to_arrays(rgeom)
    return idx, rup_meshes


def get_trt(srcs_info, srcs_grps, rups_data, idx):
    """
    :returns: the tectonic region type of the source that generated
        the rupture at position `idx`
    """
    src_info = srcs_info[rups_data[idx]['source_id']]
    print('\nSource Information')
    print('-------------------')
    for i, name in enumerate(src_info.dtype.names):
        print(f"{name:20s}: {src_info[i]}")

    src_trt = srcs_grps[src_info['grp_id']]['trt'].decode("utf-8")
    print(f"{'trt':20s}: {src_trt}")
    return src_trt


def build_gmclt(fh1, src_trt, with_betw_ratio):
    """
    :returns: the GMC logic-tree NRML (as a string) for `src_trt`
    """
    gmclt = fh1['full_lt/gsim_lt']

    tmp = gmclt.reduce([src_trt])
    tmps = ''
    for branch in tmp.branches:
        tmp_gmm = branch.gsim

        # Check if GMM provides between and within std
        if 'Inter event' not in branch.gsim.DEFINED_FOR_STANDARD_DEVIATION_TYPES:
            if isinstance(branch.gsim, ModifiableGMPE):
                kwargs = branch.gsim.params
                kwargs['add_between_within_stds'] = {
                    'with_betw_ratio': with_betw_ratio}
                tmp_gmm = valid.modified_gsim(
                    branch.gsim.gmpe,
                    **kwargs
                )
            else:
                tmp_gmm = valid.modified_gsim(
                    branch.gsim,
                    add_between_within_stds={'with_betw_ratio': with_betw_ratio}
                )

        tmps += FMT_BRANCH.format(
            bid=branch.id,
            u_model=tmp_gmm,
            u_weight=branch.weight['weight'])

    return FMT_NRML.format(content=FMT_GMC_LT.format(trt_lab=src_trt, branches=tmps))


def build_rupture_xml(rups_data, idx, rup_meshes, code2cls):
    """
    :returns: the gridded-rupture NRML (as a string) for the rupture
        at position `idx`. Note multi-fault ruptures are not supported.
    """
    coos = ''
    for lo, la, de in zip(rup_meshes[0][0].flatten(),
                          rup_meshes[0][1].flatten(),
                          rup_meshes[0][2].flatten()):
        coos += f"{lo:.6f} {la:.6f} {de:.6f} "

    names = [cls.__name__ for cls in code2cls[rups_data[idx]['code']]]

    tmpa = FMT_RUP.format(
        mag=rups_data[idx]['mag'],
        rake=rups_data[idx]['rake'],
        dep=rups_data[idx]['hypo'][2],
        lat=rups_data[idx]['hypo'][1],
        lon=rups_data[idx]['hypo'][0],
        rtype=names[1],
        stype=names[0],
        coos=coos)
    return FMT_NRML.format(content=tmpa)


def process(cfg):
    """
    Run the full workflow given a configuration dictionary.
    """
    fname = cfg['datastore']
    find_rup_id = cfg['rupture_id']

    print('Input information')
    print('-----------------')
    print(f"Input filename: {fname}")

    # Read the rupture dataset
    fh1 = dstore_read(str(fname))
    rups_data = fh1['ruptures'][:]
    rups_geom = fh1['rupgeoms'][:]
    srcs_info = fh1['source_info'][:]
    srcs_grps = fh1['source_groups'][:]

    # For future work - code2cls is a dictionary with key and integer and value
    # a tuple specifying rupture and surface types
    code2cls = {}
    code2cls.update(BaseRupture.init())

    # Retrieve the rupture
    idx, rup_meshes = select_rupture(rups_data, rups_geom, find_rup_id)

    # Get TRT from the source
    src_trt = get_trt(srcs_info, srcs_grps, rups_data, idx)

    # Create the GMC logic tree for the scenario analysis
    tmplt = build_gmclt(fh1, src_trt, cfg['with_between_within_ratio'])

    # Build the rupture .xml
    tmp = build_rupture_xml(rups_data, idx, rup_meshes, code2cls)

    print('\nOutput files')
    print('------------')
    with open(cfg['output_rupture_xml'], 'w') as fou:
        fou.write(tmp)
    print(f"Wrote {cfg['output_rupture_xml']}")

    with open(cfg['output_gmclt_xml'], 'w') as fou:
        fou.write(tmplt)
    print(f"Wrote {cfg['output_gmclt_xml']}")


def main(config, *, rupture_id=None, verbose=False):
    """ Create a scenario rupture + gmc logic tree from a SES """
    cfg = load_config(config)
    if rupture_id is not None:
        cfg['rupture_id'] = rupture_id
    if verbose:
        print(f"Configuration: {cfg}")
    breakpoint()
    process(cfg)


main.config = 'path to the TOML configuration file'
main.rupture_id = 'override the rupture_id given in the configuration file'
main.verbose = 'print the resolved configuration before running'


if __name__ == '__main__':
    sap.run(main)
