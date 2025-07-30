
import re
import toml
import pandas as pd
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement

FMT_BSET = ".//{0:s}logicTreeBranchSet"
FMT_BRANCH = "./{0:s}logicTreeBranch"


def get_unc_model(tmps):
    if re.search('\\[', tmps):
        out = toml.loads(tmps)
    else:
        tmps = tmps.replace("\n", "")
        tmps = re.sub(" ", "", tmps)
        out = f'[{tmps}]'
    return out


def get_version(namespace):
    tmp = re.split('/', re.sub('({|})', '', namespace))
    return tmp[-1]


def get_namespace(tmps):
    match = re.search('({.*})', tmps)
    return match.group(1)


def get_apply(bset):
    apply_to = {}
    for key in bset.attrib.keys():
        if re.search('^apply', key):
            apply_to[key] = bset.attrib[key]
    return apply_to


def read(fname, lttype):
    """
    lttype is gmmlt or ssmlt
    """

    tree = ET.parse(fname)
    root = tree.getroot()
    namespace = get_namespace(root.tag)

    path_bs = FMT_BSET.format(namespace)
    path_br = FMT_BRANCH.format(namespace)
    bsets = {}
    for bset in root.findall(path_bs):
        bset_id = bset.get('branchSetID')
        utype = bset.get('uncertaintyType')
        uapply = get_apply(bset)
        branches = {}
        for branch in bset.findall(path_br):
            branch_id = branch.get('branchID')
            tmp = branch.find(f'{namespace}uncertaintyModel').text
            unc_model = get_unc_model(tmp)
            unc_weight = branch.find(f'{namespace}uncertaintyWeight').text
            assert branch_id not in branches
            branches[branch_id] = {'model': unc_model,
                                   'weight': float(unc_weight)}
        bsets[bset_id] = BSet(bset_id, utype, uapply, branches)
    if lttype == 'gmmlt':
        return GmmLt(bsets)
    if lttype == 'ssmlt':
        return SmmLt(bsets)

class SmmLt():
    """ Object with LT info and methods """

    def __init__(self, bsets):
        self.bsets = bsets

    def describe(self):
        otxt = "-- Logic Tree\n"
        print(self.bsets)
        tmp = len(self.bsets)
        otxt += f"Number of branch sets : {tmp:d}"
        print(otxt)

    @classmethod
    def from_csv(cls, fname, lttype):

        # Read .csv file
        df = pd.read_csv(fname, comment='#')

        # Process tectonic regions
        szs = df.set.unique()
        bsets = {}
        for i, sz in enumerate(szs):
            smm = 0.0
            branches = []
            uapply = ''
            for j, row in df[df.set == sz].iterrows():
                branch = get_branch_ssm(row, i, j)
                smm += row.weight
                branches.append(branch)

            # Check weights
            msg = f"Weights for source zone {sz} do not sum to 1"
            assert abs(smm-1.0) < 1e-10, msg

            # Create the branchset
            utype = row.unc_type
            bset = BSet(f'sz_{i:02d}', utype, branches=branches)
            bsets[f'bs{i}'] = bset

        return cls(bsets)

    def write(self, out_fname, lt_id='', extended=False):

        root = Element('nrml')
        logictree = SubElement(root, 'logicTree')
        logictree.set('logicTreeID', lt_id)
        for bset_key in self.bsets:

            # BranchSet
            bset = self.bsets[bset_key]
            xml_branchset = SubElement(logictree, 'logicTreeBranchSet')
            xml_branchset.set('branchSetID', bset.branchSetID)
            xml_branchset.set('uncertaintyType', bset.uncertaintyType)

            # Branches
            for branch in bset.branches:
                xml_branch = SubElement(xml_branchset, 'logicTreeBranch')
                xml_branch.set('branchID', branch['bid'])
                xml_model = SubElement(xml_branch, 'uncertaintyModel')
                xml_weight = SubElement(xml_branch, 'uncertaintyWeight')
                xml_weight.text = '{:.5f}'.format(branch['weight'])
                xml_model.text = branch['sources']

        tree = ET.ElementTree(root)
        ET.indent(tree, space="\t", level=0)
        tree.write(out_fname, encoding="UTF-8", xml_declaration=True)


class GmmLt():
    """ Object with LT info and methods """

    def __init__(self, bsets):
        self.bsets = bsets

    def describe(self):
        otxt = "-- Logic Tree\n"
        print(self.bsets)
        tmp = len(self.bsets)
        otxt += f"Number of branch sets : {tmp:d}"
        print(otxt)

    @classmethod
    def from_csv(cls, fname):

        # Read .csv file
        df = pd.read_csv(fname, comment='#')

        # Process tectonic regions
        trts = df.tectonic_region.unique()
        bsets = {}
        for i, trt in enumerate(trts):
            smm = 0.0
            branches = []
            uapply = ''
            for j, row in df[df.tectonic_region == trt].iterrows():
                branch = get_branch(row, i, j)
                smm += row.weight
                branches.append(branch)

            # Check weights
            msg = f"Weights for tectonic region {trt} do not sum to 1"
            assert abs(smm-1.0) < 1e-10, msg

            # Create the branchset
            uapply = {'applyToTectonicRegionType': trt}
            bset = BSet(f'bs_{i:02d}', 'gmpeModel', uapply, branches=branches)
            bsets[f'bs{i}'] = bset

        return cls(bsets)

    def write(self, out_fname, lt_id='', extended=False):

        root = Element('nrml')
        logictree = SubElement(root, 'logicTree')
        logictree.set('logicTreeID', lt_id)
        for bset_key in self.bsets:

            # BranchSet
            bset = self.bsets[bset_key]
            xml_branchset = SubElement(logictree, 'logicTreeBranchSet')
            xml_branchset.set('branchSetID', bset.branchSetID)
            xml_branchset.set('uncertaintyType', bset.uncertaintyType)
            for app_key in bset.uapply:
                xml_branchset.set(app_key, bset.uapply[app_key])

            # Branches
            for branch in bset.branches:
                xml_branch = SubElement(xml_branchset, 'logicTreeBranch')
                xml_branch.set('branchID', branch['bid'])
                xml_model = SubElement(xml_branch, 'uncertaintyModel')
                if len(branch['adjust']):
                    txt = '\n[ModifiableGMPE]\n'
                    tmps = '{}'
                    ext = False
                    if len(branch['params']) > 0:
                        prs, ext = get_param_str(branch, extended)
                    txt += 'gmpe.{:s} = {}\n'.format(branch['model'], tmps)
                    if ext:
                        txt += ext
                    for adj_key in branch['adjust']:
                        if len(branch['adjust'][adj_key]) < 1:
                            txt += '{:s} = {{}}\n'.format(adj_key)
                        for par_key in branch['adjust'][adj_key]:

                            tmpv = branch['adjust'][adj_key][par_key]

                            if isinstance(tmpv, str):
                                fmt = "{:s}.{:s} = '{}'\n"
                            else:
                                fmt = "{:s}.{:s} = {}\n"

                            txt += fmt.format(adj_key, par_key, tmpv)

                xml_model.text = txt
                xml_weight = SubElement(xml_branch, 'uncertaintyWeight')
                xml_weight.text = '{:.5f}'.format(branch['weight'])

        tree = ET.ElementTree(root)
        ET.indent(tree, space="\t", level=0)
        tree.write(out_fname, encoding="UTF-8", xml_declaration=True)


def get_param_str(branch, newline=False, extended=False):

    ext = ''
    tmps = '{'
    gmm = branch['model']
    for i, key in enumerate(branch['params']):

        if i > 0:
            tmps += ', '

        if (isinstance(branch['params'][key], str) and
                re.search('[a-zA-Z]', branch['params'][key])):
            if re.search('^\\{', branch['params'][key]):
                tmp = branch['params'][key]
                tmp = re.sub(';', ',', tmp)
                tmp = re.sub(' ', '', tmp)
                branch['params'][key] = tmp
            # String
            tmps += "{:s} = '{}'".format(key, branch['params'][key])
            if re.search('(\\[|\\]|\\{)', branch['params'][key]):
                ext += "gmpe.{:s}.{:s} = {}\n".format(gmm, key,
                                                      branch['params'][key])
            else:
                ext += "gmpe.{:s}.{:s} = \"{}\"\n".format(gmm, key,
                                                      branch['params'][key])
        else:
            # Number
            tmps += "{:s} = {}".format(key, branch['params'][key])
            ext += "gmpe.{:s}.{:s} = {}\n".format(gmm, key,
                                                branch['params'][key])

    tmps += '}'

    if extended:
        tmps = '{}'

    return tmps, ext


class BSet():
    """
    Object containing information about a branchset

    :param bsid:
        ID of the branch set
    :param uncertaintyType:
        The type of uncertainty modelled
    :param uapply:
        A dictionaty with key the apply rule
    :param branches:
        A dictionaty with key the branch ID and value an dictionary with two
        keys: 'model' and 'weight'
    """

    def __init__(self, bsid, utype, uapply=None, branches=None):
        self.branchSetID = bsid
        self.uncertaintyType = utype
        self.uapply = uapply
        self.branches = branches

    def describe(self):
        otxt = "-- Branch Set\n"
        otxt += f"Uncertainty type      : {self.utype}\n"
        otxt += f"Number of branches    : {len(self.branches)}\n"
        print(otxt)


def get_branch(row, i, j):
    """
    :param row:
        A :class:`pandas.Series` instance
    """

    # Set branch ID
    if 'branch_id' not in row:
        bid = f'b_{i:02d}_{j:02d}'
    else:
        bid = row.branch_id

    # Read adjustements and parameters
    adjusts_vals = get_data(row, 'adjust')
    params_vals = get_data(row, 'param')

    return {'bid': bid, 'model': row.model, 'params': params_vals,
            'adjust': adjusts_vals, 'weight': row.weight}

def get_branch_ssm(row, i, j):
    """
    :param row:
        A :class:`pandas.Series` instance
    """

    # Set branch ID
    if 'branch_id' not in row:
        bid = f'b_{i:02d}_{j:02d}'
    else:
        bid = row.branch_id


    return {'bid': bid, 'sources': row.sources, 'weight': row.weight}



def get_data(row, prefix):
    vals = {}
    pattern = '^{:s}_(.*)'.format(prefix)
    pattern_var = 'var_(.*)'
    input_keys = list(row.keys())

    # Process keys
    for i, key in enumerate(input_keys):

        # Get the method from the header
        m0 = re.search(pattern, key)
        if m0:

            # This sets parameters assigned directly to the GMM model when
            # instantiated (they are defined with the prefix 'param_') and
            # collects the variables ('var_') required for a given modification
            # (in the .csv this is specified with the prefix adjust_)
            if not isinstance(row.loc[key], bool):
                if not pd.isna(row.loc[key]):
                    vals[m0.group(1)] = row.loc[key]
            elif row.loc[key]:
                j = i + 1
                params = {}
                while (j < len(input_keys) and
                       re.search(pattern_var, input_keys[j])):
                    m = re.search(pattern_var, input_keys[j])
                    params[m.group(1)] = row.loc[input_keys[j]]
                    j += 1
                vals[m0.group(1)] = params

    return vals
