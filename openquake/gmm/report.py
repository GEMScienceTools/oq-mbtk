""" Module :mod:`openquake.gmm.report` """

import re

from prettytable import PrettyTable

from openquake.hazardlib import gsim
from openquake.gmm.matrix import list_2_str
from openquake.commonlib.logictree import GsimLogicTree

GMPES = gsim.get_available_gsims()


class GMC:
    """
    :param str fname:
        The name of the nrml formatted ground motion logic tree
    :param list tr_types:
        A list of strings each one specifying a tectonic region type
    """

    def __init__(self, fname, tr_types=['*']):
        gsim_lt = GsimLogicTree(fname, tr_types)
        self.gsims_by_trt = gsim_lt.values

    def write_table(self, fname):
        fo = open(fname, 'w')
        for trt in sorted(self.gsims_by_trt):
            fo.write('-- {:s}\n'.format(trt))
            fo.write(self._get_txt_trt_table(trt).get_string())
            fo.write('\n')
        fo.close()

    def _get_txt_trt_table(self, trt):
        x = PrettyTable()
        x.field_names = ["Name", "Site"]
        for gs in self.gsims_by_trt[trt]:
            tmps = list_2_str(list(gs.REQUIRES_SITES_PARAMETERS))
            x.add_row([str(gs), tmps])
        return x


class GMCcomp:
    """
    :param list fname_list:
        A list
    :param list tr_types:
    """

    def __init__(self, fname_list, tr_types=['*']):
        self.fname_list = fname_list
        self.gsims_by_trt_list = []
        self.trts = set()
        self.gsims = {}
        # update the list of dictionaries
        for fname in fname_list:
            gsim_lt = GsimLogicTree(fname, tr_types)
            self.gsims_by_trt_list.append(gsim_lt.values)
        # update the set of TRT
        self._set_trts()
        self._set_gsim_inv()
        self._set_gsim_inv_trt()

    def _set_trts(self):
        for gsim_dict in self.gsims_by_trt_list:
            for key in gsim_dict:
                self.trts |= set([key])

    def _set_gsim_inv(self):
        for i, gsim_dict in enumerate(self.gsims_by_trt_list):
            lab = '{:d}'.format(i)
            for trt in gsim_dict:
                for gslab in gsim_dict[trt]:
                    tmps = str(gslab)
                    if tmps not in self.gsims:
                        self.gsims[tmps] = set([lab])
                    else:
                        self.gsims[tmps] |= set([lab])

    def _set_gsim_inv_trt(self):
        """
        Create an inverted file structure for the GMPEs in each tectonic
        region.
        """
        dct = {}
        for i, gsim_dict in enumerate(self.gsims_by_trt_list):
            model_lab = '{:d}'.format(i)
            for trt in gsim_dict:
                if trt not in dct:
                    dct[trt] = {}
                for gslab in gsim_dict[trt]:
                    tmps = re.sub('(\(|\))', '', str(gslab))
                    if tmps not in dct[trt]:
                        dct[trt][tmps] = set([model_lab])
                    else:
                        dct[trt][tmps] |= set([model_lab])
        self.gsim_inv = dct

    def write_gmsim_summary(self, fname, attrs=[]):
        """
        """
        x = PrettyTable()
        names = ["GMPE"]
        names += (['{:d}'.format(i) for i in range(len(self.fname_list))])
        if len(attrs):
            names += attrs
        x.field_names = names
        for gslab in sorted(self.gsims.keys()):
            tmpl = [gslab]
            for i in range(len(self.fname_list)):
                lab = '{:d}'.format(i)
                if lab in self.gsims[gslab]:
                    tmpl.append('x')
                else:
                    tmpl.append(' ')
            gs = GMPES[re.sub('(\\(|\\)|\\[|\\])', '', gslab)]()
            for attr in attrs:
                tmpl.append(list_2_str((list(getattr(gs, attr)))))
            x.add_row(tmpl)
        fo = open(fname, 'w')
        fo.write(x.get_string())
        fo.close()

    def write_gmsim_summary_per_trt(self, fname, attrs=[], mlabs=[]):
        """
        """
        maxlen = 0
        for trt in sorted(self.gsim_inv.keys()):
            maxlen = max(maxlen, len(trt))
        # Create the table
        x = PrettyTable()
        names = ["GMPE"]
        if len(mlabs) < 1:
            names += (['{:d}'.format(i) for i in range(len(self.fname_list))])
        else:
            names += (['{:s}'.format(l) for l in mlabs])
        names += ['Total']
        if len(attrs):
            names += attrs
        x.field_names = names
        x.align["GMPE"] = "l"
        # Populate the table
        for trt in sorted(self.gsim_inv.keys()):
            # This creates the trt header
            tmpl = [trt.upper().rjust(maxlen+5, ' ')]
            for i, gsim_dict in enumerate(self.gsims_by_trt_list):
                model_lab = '{:d}'.format(i)
                tmpl.append('----')
            tmpl.append(' ')
            for i in range(len(attrs)):
                tmpl.append('------------')
            x.add_row(tmpl)
            # Now work on gsim
            for gsim_key in sorted(self.gsim_inv[trt]):
                tmpl = [gsim_key]
                cnt = 0
                for i, gsim_dict in enumerate(self.gsims_by_trt_list):
                    model_lab = '{:d}'.format(i)
                    if model_lab in self.gsim_inv[trt][gsim_key]:
                        cnt += 1
                        tmpl.append('x')
                    else:
                        tmpl.append(' ')
                tmpl.append('{:d}'.format(cnt))
                gs = GMPES[re.sub('(\\(|\\)|\\[|\\])', '', gsim_key)]()
                for attr in attrs:
                    tmpl.append(list_2_str((list(getattr(gs, attr)))))
                x.add_row(tmpl)
        fo = open(fname, 'w')
        fo.write(x.get_string())
        fo.close()
