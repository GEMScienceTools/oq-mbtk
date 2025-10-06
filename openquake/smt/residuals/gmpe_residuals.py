# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2025 GEM Foundation and G. Weatherill
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.
"""
Module to get GMPE residuals - total, inter and intra
{'GMPE': {'IMT1': {'Total': [], 'Inter event': [], 'Intra event': []},
          'IMT2': { ... }}}
"""
import sys
import warnings
import copy
import re
import toml
import numpy as np
import pandas as pd
from math import sqrt, ceil
from scipy.integrate import trapezoid
from scipy.special import erf
from scipy.stats import norm

from openquake.hazardlib import valid
from openquake.hazardlib import imt
from openquake.smt.residuals.sm_database_selector import SMRecordSelector
from openquake.smt.utils import convert_accel_units, check_gsim_list


ALL_SIGMA = frozenset({'Inter event', 'Intra event', 'Total'})

RUP_PAR = ["mag",
           "strike",
           "dip",
           "rake",
           "ztor",
           "width",
           "hypo_lon",
           "hypo_lat",
           "hypo_depth"]

ST_PAR = ["vs30",
          "custom_site_id",
          "lons",
          "lats",
          "depths",
          "z1pt0",
          "z2pt5",
          "rrup",
          "rx",
          "rjb",
          "rhypo",
          "repi",
          "ry0"]


### Util functions
def get_gmm_from_toml(key, config):
    """
    Get a GMM from a TOML file
    """
    # ModifiableGMPE is not implemented for use in res module
    if key == "ModifiableGMPE":
        raise ValueError("The use of ModifiableGMPE is not"
        "supported within the residuals module.")

    # If the key contains a number we take the second part
    if re.search("^\\d+\\-", key):
        tmp = re.sub("^\\d+\\-", "", key)
        value = f"[{tmp}] "
    else:
        value = f"[{key}] "
    if len(config['models'][key]):
        config['models'][key].pop('style', None)
        value += '\n' + str(toml.dumps(config['models'][key]))
    return valid.gsim(value.strip())

def get_gmpe_str(gmpe):
    """
    Return a string of the GMPE to use for printing/exporting
    """
    if '_toml=' in str(gmpe):
        gmpe_str = str(
            gmpe).split('_toml=')[1].replace(')','').replace('\n','; ')
    else:
        gmpe_str = gmpe

    return gmpe_str


class Residuals(object):
    """
    Residuals object for storing ground-motion residuals computed
    for a given list of GMMs and IMTs.
    """
    def __init__(self, gmpe_list, imts):
        """
        :param  gmpe_list:
            A list e.g. ['BooreEtAl2014', 'CauzziEtAl2014']
        :param  imts:
            A list e.g. ['PGA', 'SA(0.1)', 'SA(1.0)']
        """
        # Residuals object
        gmpe_list = copy.deepcopy(gmpe_list)
        self.gmpe_list = check_gsim_list(gmpe_list)
        self.types = {gmpe: {} for gmpe in self.gmpe_list}
        self.residuals = []
        self.modelled = []
        self.imts = imts
        self.unique_indices = {}
        self.gmpe_sa_limits = {}
        self.gmpe_scalars = {}
        for gmpe in self.gmpe_list:
            gmpe_dict_1 = {}
            gmpe_dict_2 = {}
            self.unique_indices[gmpe] = {}
            
            # Get the period range and the coefficient types
            gmpe_i = self.gmpe_list[gmpe]
            coeff_atts = [att for att in dir(gmpe_i) if "COEFFS" in att]
            if len(coeff_atts) > 0:
                coeff_att = coeff_atts[0] # Some GSIMS have irreg. COEFF attribute 
                                          # names e.g. Z06 (but const. period range)
                pers = [sa.period for sa in getattr(gmpe_i, coeff_att).sa_coeffs]
                self.gmpe_scalars[gmpe] = list(
                    getattr(gmpe_i, coeff_att).non_sa_coeffs)
            else:
                assert hasattr(gmpe_i, "gmpe_table")
                # tabular GMM specified using an alias
                pers = gmpe_i.imls["T"]

            min_per, max_per = (min(pers), max(pers))
            self.gmpe_sa_limits[gmpe] = (min_per, max_per)
            for imtx in self.imts:
                if "SA(" in imtx:
                    period = imt.from_string(imtx).period
                    if period < min_per or period > max_per:
                        print(f"IMT {imtx} outside period range for GMPE {gmpe}"
                              f"(min GMM period = {min_per} s, "
                              f"max GMM period = {max_per} s)")
                        gmpe_dict_1[imtx] = None
                        gmpe_dict_2[imtx] = None
                        continue
                gmpe_dict_1[imtx] = {}
                gmpe_dict_2[imtx] = {}
                self.unique_indices[gmpe][imtx] = []
                self.types[gmpe][imtx] = []

                # If mixed effects GMPE fix res_type order
                if gmpe_i.DEFINED_FOR_STANDARD_DEVIATION_TYPES == ALL_SIGMA:
                    for res_type in ['Total','Inter event', 'Intra event']:
                        gmpe_dict_1[imtx][res_type] = []
                        gmpe_dict_2[imtx][res_type] = []
                        self.types[gmpe][imtx].append(res_type)
                    gmpe_dict_2[imtx]["Mean"] = []
           
                # For handling of GMPEs with total sigma only
                else:
                    for res_type in gmpe_i.DEFINED_FOR_STANDARD_DEVIATION_TYPES:
                        gmpe_dict_1[imtx][res_type] = []
                        gmpe_dict_2[imtx][res_type] = []
                        self.types[gmpe][imtx].append(res_type)
                    gmpe_dict_2[imtx]["Mean"] = []
            
            self.residuals.append([gmpe, gmpe_dict_1])
            self.modelled.append([gmpe, gmpe_dict_2])

        self.residuals = dict(self.residuals)
        self.modelled = dict(self.modelled)
        self.number_records = None
        self.contexts = None

    @classmethod
    def from_toml(cls, filename):
        """
        Read in gmpe_list and imts from .toml file. This method allows use of
        gmpes with additional parameters and input files within the SMT
        """
        # Read in toml file with dict of gmpes and subdict of imts
        config = toml.load(filename)

        # Parsing file with models
        gmpe_list = []
        for _, key in enumerate(config['models']):
            
            # Get toml representation of GMM
            gmm = get_gmm_from_toml(key, config)
            
            # Create valid gsim object
            gmpe_list.append(gmm)
            
        # Get imts    
        imts = config['imts']['imt_list']     
        
        return cls(gmpe_list, imts)

    def compute_residuals(self,
                          ctx_database,
                          nodal_plane_index=1,
                          component="Geometric",
                          normalise=True):
        """
        Calculate the residuals for a set of ground motion records

        :param ctx_database: a :class:`context_db.ContextDB`, i.e. a database of
            records capable of returning dicts of earthquake-based Contexts and
            observed IMTs.
            See e.g., :class:`openquake.smt.sm_database.GroundMotionDatabase`
            for an example
        """
        # Build initial contexts with the observed values
        contexts = ctx_database.get_contexts(
            nodal_plane_index, self.imts, component)
        
        # Check at least one observed value per IMT (else raise an error)
        for im in self.imts:
            obs_check = []
            for ctx in contexts:
                obs_check.append(ctx["Observations"][im])
            obs_check = np.concatenate(obs_check)
            check = pd.notnull(obs_check)
            if len(check[check]) < 1:
                raise ValueError(f"All observed intensity measure "
                                 f"levels for {im} are empty - "
                                 f"no residuals can be computed "
                                 f"for {im}")

        # Get IMTs which need acc. units conv. from cm/s^2 to g
        accel_imts = tuple(
            [imtx for imtx in self.imts if (imtx == "PGA" or "SA(" in imtx)])

        # Contexts is in either case a list of dictionaries
        self.contexts = []
        for context in contexts:
            # If no rvolc fix to zero (ensure rvolc gsims usable)
            if 'rvolc' not in context['Ctx']._slots_:
                context['Ctx'].rvolc = np.zeros_like(context['Ctx'].repi)
            # Convert all IMTS with acceleration units, which are supposed to
            # be in cm/s/s, to g:
            for a_imt in accel_imts:
                context['Observations'][a_imt] = convert_accel_units(
                        context['Observations'][a_imt], 'cm/s/s', 'g')
            # Get the expected ground motions from GMMs
            context = self.get_exp_motions(context)
            context = self.calculate_residuals(context, normalise)
            for gmpe in self.residuals.keys():
                for imtx in self.residuals[gmpe].keys():
                    if not context["Residual"][gmpe][imtx]:
                        continue
                    for res_type in self.residuals[gmpe][imtx].keys():
                        if res_type == "Inter event":
                            inter_ev = context["Residual"][gmpe][imtx][res_type]
                            if len(inter_ev) < 1:
                                # Dummy to pass first conditional with indexing
                                # if no obs values for given IMT for the event
                                inter_ev = np.array([np.nan]) 
                            if np.all(np.fabs(inter_ev - inter_ev[0])
                                      < 1.0E-12):
                                # Single inter-event residual
                                self.residuals[gmpe][imtx][res_type].append(
                                    inter_ev[0])
                                # Append indices
                                self.unique_indices[gmpe][imtx].append(
                                    np.array([0]))
                            else:
                                # Inter event residuals per-site e.g. Chiou
                                # & Youngs (2008; 2014) case
                                self.residuals[gmpe][imtx][res_type].extend(
                                    inter_ev)
                                self.unique_indices[gmpe][imtx].append(
                                    np.arange(len(inter_ev)))
                        else:
                            self.residuals[gmpe][imtx][res_type].extend(
                                context["Residual"][gmpe][imtx][res_type])
                        self.modelled[gmpe][imtx][res_type].extend(
                            context["Expected"][gmpe][imtx][res_type])

                    self.modelled[gmpe][imtx]["Mean"].extend(
                        context["Expected"][gmpe][imtx]["Mean"])

            self.contexts.append(context)

        for gmpe in self.residuals.keys():
            for imtx in self.residuals[gmpe].keys():
                # Check residuals exist for GMM and IMT
                if not self.residuals[gmpe][imtx]:
                    continue
                for res_type in self.residuals[gmpe][imtx].keys():
                    self.residuals[gmpe][imtx][res_type] = np.array(
                        self.residuals[gmpe][imtx][res_type])
                    self.modelled[gmpe][imtx][res_type] = np.array(
                        self.modelled[gmpe][imtx][res_type])
                self.modelled[gmpe][imtx]["Mean"] = np.array(
                    self.modelled[gmpe][imtx]["Mean"])

    def get_exp_motions(self, context):
        """
        Calculate the expected ground motions from the context
        """
        # Get expected
        exp = {gmpe: {} for gmpe in self.gmpe_list}
        # Period range for GSIM
        for _, gmpe in enumerate(self.gmpe_list):
            exp[gmpe] = {imtx: {} for imtx in self.imts}
            for imtx in self.imts:
                gsim = self.gmpe_list[gmpe]
                if "SA(" in imtx:
                    period = imt.from_string(imtx).period
                    if (period < self.gmpe_sa_limits[gmpe][0] or
                        period > self.gmpe_sa_limits[gmpe][1]):
                        exp[gmpe][imtx] = None
                        continue
                # Get expected motions
                mean, stddev = gsim.get_mean_and_stddevs(
                    context["Ctx"],
                    context["Ctx"],
                    context["Ctx"],
                    imt.from_string(imtx),
                    self.types[gmpe][imtx])
                keep = context["Retained"][imtx]
                mean = mean[keep]
                for idx_comp, comp in enumerate(stddev):
                    stddev[idx_comp] = comp[keep]
                # If no sigma for the GMM residuals can't be computed
                if np.all(stddev[0] == 0.) and len(keep) > 0:
                    gs = str(gmpe).split('(')[0]
                    m = 'A sigma model is not provided for %s' %gs
                    raise ValueError(m)
                exp[gmpe][imtx]["Mean"] = mean
                for i, res_type in enumerate(self.types[gmpe][imtx]):
                    exp[gmpe][imtx][res_type] = stddev[i]

        context["Expected"] = exp

        return context

    def calculate_residuals(self, context, normalise=True):
        """
        Calculate the residual terms
        """
        # Calculate residual
        residual = {}
        for gmpe in self.gmpe_list:
            residual[gmpe] = {}
            for imtx in self.imts:
                residual[gmpe][imtx] = {}
                obs = np.log(context["Observations"][imtx])
                keep = context["Retained"][imtx]
                obs = obs[keep]
                if not context["Expected"][gmpe][imtx]:
                    residual[gmpe][imtx] = None
                    continue
                mean = context["Expected"][gmpe][imtx]["Mean"]
                total_stddev = context["Expected"][gmpe][imtx]["Total"]
                residual[gmpe][imtx]["Total"] = (obs - mean) / total_stddev
                if "Inter event" in self.residuals[gmpe][imtx].keys():
                    inter, intra = self._get_random_effects_residuals(
                        obs,
                        mean,
                        context["Expected"][gmpe][imtx]["Inter event"],
                        context["Expected"][gmpe][imtx]["Intra event"],
                        normalise)
                    residual[gmpe][imtx]["Inter event"] = inter
                    residual[gmpe][imtx]["Intra event"] = intra
        context["Residual"] = residual
        
        return context

    def _get_random_effects_residuals(self,
                                      obs,
                                      mean,
                                      inter,
                                      intra,
                                      normalise=True):
        """
        Calculates the random effects residuals using the inter-event
        residual formula described in Abrahamson & Youngs (1992) Eq. 10
        
        :param obs: array of observed ground-shaking values for a single ctx
                    (i.e. event) for a given imt, in natural log.
        :param mean: array of ground-shaking values for the same ctx 
                     predicted by the given GMPE and imt, in natural log.
        :param inter: float representing the inter-event component of GMPE
                      sigma for a given imt.
        :param intra: float representing the intra-event component of GMPE
                      sigma for a given imt.
        :param normalise: bool which if True normalises the residuals using
                          the corresponding GMPE sigma components
        """
        nvals = float(len(mean))
        v = nvals * (inter ** 2.) + (intra ** 2.)
        inter_res = ((inter ** 2.) * sum(obs - mean)) / v
        intra_res = obs - (mean + inter_res)
        if normalise:
            return inter_res / inter, intra_res / intra
        else:
            return inter_res, intra_res

    def get_residual_statistics(self):
        """
        Retreives the mean and standard deviation values of the residuals
        """
        statistics = {gmpe: {} for gmpe in self.gmpe_list}
        for gmpe in self.gmpe_list:
            for imtx in self.imts:
                if not self.residuals[gmpe][imtx]:
                    continue
                statistics[
                    gmpe][imtx] = self.get_residual_statistics_for(gmpe, imtx)
                
        return statistics

    def get_residual_statistics_for(self, gmpe, imtx):
        """
        Retreives the mean and standard deviation values of the residuals for
        a given gmpe and imtx

        :param gmpe: (string) the gmpe. It must be in the list of this
            object's gmpes
        :param imtx: (string) the imt. It must be in the imts defined for
            the given `gmpe`
        """
        residuals = self.residuals[gmpe][imtx]
        return {
            res_type: {
                "Mean": np.nanmean(residuals[res_type]),
                "Std Dev": np.nanstd(residuals[res_type])
                } for res_type in self.types[gmpe][imtx]}

    def _get_magnitudes(self):
        """
        Returns an array of magnitudes equal in length to the number of
        residuals
        """
        magnitudes = np.array([])
        for ctxt in self.contexts:
            magnitudes = np.hstack([
                magnitudes,
                ctxt["Ctx"].mag * np.ones(len(ctxt["Ctx"].repi))])
            
        return magnitudes

    def export_residuals(self, out_fname):
        """
        Export the observed, predicted and residuals to a text file
        """
        ctxs = self.contexts # List of contexts
        gmms = self.gmpe_list
        imts = self.imts
        store = {}
        for ctx in ctxs:
            for imt in imts:
                ctx_and_imt = {} # One df per imt and ctx
                for gmpe in gmms:                
                    gmpe_str = get_gmpe_str(gmpe)

                    # Get the expected values and the residuals
                    res = ctx["Residual"][gmpe][imt]
                    exp = ctx["Expected"][gmpe][imt]
                    for comp in res:
                        
                        # Make a key
                        key = f"GMM={gmpe_str}_IMT={imt}_{comp}"
                        key = key.replace(" ", "_")
                        key = key.replace(";", "")

                        # Store each set of values
                        ctx_and_imt[key+"_Residuals"] = res[comp]
                        ctx_and_imt[key+"_Predicted"] = exp[comp]

                # Get observed with the NaNs (empty recs for IMT) removed
                obs = ctx["Observations"][imt]
                keep = ctx["Retained"][imt]
                key_obs = f"IMT={imt}_Observations"
                ctx_and_imt[key_obs] = obs[keep]

                # Get the event info
                for par in RUP_PAR:
                    val = np.full(len(keep), getattr(ctx["Ctx"], par))
                    ctx_and_imt[par] = val

                # Get the station info
                for par in ST_PAR:
                    val = np.array(getattr(ctx["Ctx"], par))[keep]
                    ctx_and_imt[par] = val

                # Into a dataframe and rename some columns
                ctx_df = pd.DataFrame(ctx_and_imt)
                ctx_df = ctx_df.rename(columns={"custom_site_id": "st_code",
                                                "lons": "st_lon",
                                                "lats": "st_lat",
                                                "depths": "st_elevation"})

                # Store the DataFrame for the event
                store[f"{ctx['EventID']}_IMT={imt}"] = ctx_df

        # Now write results for the event to a text file
        with open(out_fname, 'w') as f:
            for ev_imt, ev_imt_df in store.items():
                ev = ev_imt.split("IMT")[0][:-1]
                imt = ev_imt.split("IMT=")[1]
                f.write(f"Event:{ev} IMT: {imt}\n")
                f.write(ev_imt_df.to_string(index=False))
                f.write("\n\n")

    ### Likelihood (Scherbaum et al. 2004) functions
    def get_likelihood_values(self):
        """
        Returns the likelihood values for Total, plus inter- and intra-event
        residuals according to Equation 9 of Scherbaum et al (2004)
        """
        statistics = self.get_residual_statistics()
        lh_values = {gmpe: {} for gmpe in self.gmpe_list}
        for gmpe in self.gmpe_list:
            for imtx in self.imts:
                # Check residuals exist for GMM and IMT
                if not self.residuals[gmpe][imtx]:
                    print("IMT %s not found in Residuals for %s"
                          % (imtx, gmpe))
                    continue
                lh_values[gmpe][imtx] = {}
                values = self._compute_likelihood_values_for(gmpe, imtx)
                for res_type, data in values.items():
                    l_h, median_lh = data
                    lh_values[gmpe][imtx][res_type] = l_h
                    statistics[gmpe][imtx][res_type]["Median LH"] = median_lh

        return lh_values, statistics

    def _compute_likelihood_values_for(self, gmpe, imt):
        """
        Returns the likelihood values for Total, plus inter- and intra-event
        residuals according to Equation 9 of Scherbaum et al (2004) for the
        given gmpe and the given intensity measure type.
        `gmpe` must be in this object gmpe(s) list and imt must be defined
        for the given gmpe: this two conditions are not checked for here.

        :return: a dict mapping the residual type(s) (string) to the tuple
        lh, median_lh where the first is the array of likelihood values and
        the latter is the median of those values
        """
        ret = {}
        for res_type in self.types[gmpe][imt]:
            zvals = np.fabs(self.residuals[gmpe][imt][res_type])
            l_h = 1.0 - erf(zvals / sqrt(2.))
            median_lh = np.nanpercentile(l_h, 50.0)
            ret[res_type] = l_h, median_lh

        return ret

    ### LLH (Scherbaum et al. 2009) functions
    def get_loglikelihood_values(self):
        """
        Returns the loglikelihood fit of the GMPEs to data using the
        loglikehood (LLH) function described in Scherbaum et al. (2009).
        
        Scherbaum, F., Delavaud, E., Riggelsen, C. (2009) "Model Selection in
        Seismic Hazard Analysis: An Information-Theoretic Perspective",
        Bulletin of the Seismological Society of America, 99(6), 3234-3247

        :param imts:
            List of intensity measures for LLH calculation
        """
        imt_list = {imt: None for imt in self.imts}
        imt_list["All"] = None
        self.llh = {gmpe: {} for gmpe in self.gmpe_list}
        for gmpe in self.gmpe_list:
            log_residuals = np.array([])
            for imtx in self.imts:
            
                # Check residuals exist for GMM and IMT
                if not (imtx in self.imts) or not self.residuals[gmpe][imtx]:
                    print("IMT %s not found in Residuals for %s" % (imtx, gmpe))
                    continue
            
                # Get log-likelihood distance for IMT
                asll = np.log2(
                    norm.pdf(self.residuals[gmpe][imtx]["Total"], 0., 1.0))
                self.llh[gmpe][imtx] = -(1.0 / float(len(asll))) * np.sum(asll)
            
                # Stack
                log_residuals = np.hstack([log_residuals, asll])

            # Get the average over the IMTs
            self.llh[gmpe]["All"] = -(
                1. / float(len(log_residuals))) * np.sum(log_residuals)

        # Get mean weights
        weights = np.array(
            [2.0 ** -self.llh[gmpe]["All"] for gmpe in self.gmpe_list])
        weights = weights / np.sum(weights)
        self.model_weights = {
            gmpe: weights[idx] for idx, gmpe in enumerate(self.gmpe_list)}

        # Get weights with imt
        self.model_weights_with_imt = {}
        for im in self.imts:
            weights_with_imt = np.array(
                [2.0 ** -self.llh[gmpe][im] for gmpe in self.gmpe_list])
            weights_with_imt = weights_with_imt/np.sum(weights_with_imt)
            self.model_weights_with_imt[im] = {gmpe: weights_with_imt[
                idx] for idx, gmpe in enumerate(self.gmpe_list)}
            
        return self.llh, self.model_weights, self.model_weights_with_imt

    ### EDR (Kale and Akkar 2013) functions
    def get_edr_values(self, bandwidth=0.01, multiplier=3.0):
        """
        Calculates the EDR values for each GMPE according to the Euclidean
        Distance Ranking method of Kale & Akkar (2013):

        Kale, O., and Akkar, S. (2013) "A New Procedure for Selecting and
        Ranking Ground Motion Predicion Equations (GMPEs): The Euclidean
        Distance-Based Ranking Method", Bulletin of the Seismological Society
        of America, 103(2A), 1069 - 1084.

        :param float bandwidth:
            Discretisation width

        :param float multiplier:
            "Multiplier of standard deviation (equation 8 of Kale and Akkar)
        """
        edr_values = {gmpe: {} for gmpe in self.gmpe_list}
        for gmpe in self.gmpe_list:
            obs, exp, std = self._get_edr_inputs(gmpe)
            results = self._compute_edr(obs,
                                        exp,
                                        std,
                                        bandwidth,
                                        multiplier)
            edr_values[gmpe]["MDE Norm"] = results[0]
            edr_values[gmpe]["sqrt Kappa"] = results[1]
            edr_values[gmpe]["EDR"] = results[2]

        return edr_values
    
    def get_edr_values_wrt_imt(self, bandwidth=0.01, multiplier=3.0):
        """
        Calculates the EDR values for each GMPE according to the Euclidean
        Distance Ranking method of Kale & Akkar (2013) for each imt

        Kale, O., and Akkar, S. (2013) "A New Procedure for Selecting and
        Ranking Ground Motion Predicion Equations (GMPEs): The Euclidean
        Distance-Based Ranking Method", Bulletin of the Seismological Society
        of America, 103(2A), 1069 - 1084.

        :param float bandwidth:
            Discretisation width

        :param float multiplier:
            "Multiplier of standard deviation (equation 8 of Kale and Akkar)
        """
        self.edr_values_wrt_imt = {gmpe: {} for gmpe in self.gmpe_list}
        for gmpe in self.gmpe_list:
            obs_wrt_imt, exp_wrt_imt, std_wrt_imt = self._get_edr_inputs_wrt_imt(gmpe)
            results = self._compute_edr_wrt_imt(obs_wrt_imt,
                                                exp_wrt_imt,
                                                std_wrt_imt,
                                                bandwidth,
                                                multiplier)
            self.edr_values_wrt_imt[gmpe]["MDE Norm"] = results[0]
            self.edr_values_wrt_imt[gmpe]["sqrt Kappa"] = results[1]
            self.edr_values_wrt_imt[gmpe]["EDR"] = results[2]

        return self.edr_values_wrt_imt

    def _get_edr_inputs(self, gmpe):
        """
        Extract the observed ground motions, expected and total standard
        deviation for the GMPE
        """
        obs = np.array([], dtype=float)
        exp = np.array([], dtype=float)
        std = np.array([], dtype=float)
        for imtx in self.imts:
            for context in self.contexts:
                keep = context["Retained"][imtx]
                obs = np.hstack(
                    [obs, np.log(context["Observations"][imtx][keep])])
                exp = np.hstack([exp, context["Expected"][gmpe][imtx]["Mean"]])
                std = np.hstack([std, context["Expected"][gmpe][imtx]["Total"]])

        return obs, exp, std
    
    def _get_edr_inputs_wrt_imt(self, gmpe):
        """
        Extract the observed ground motions, expected and total standard
        deviation for the GMPE (per imt)
        """  
        # Get EDR values per imt
        obs_wrt_imt, exp_wrt_imt, std_wrt_imt = {}, {}, {}
        for imtx in self.imts:
            obs = np.array([], dtype=float)
            exp = np.array([], dtype=float)
            std = np.array([], dtype=float)
            for context in self.contexts:
                keep = context["Retained"][imtx]
                obs_stack = np.log(context["Observations"][imtx][keep])
                obs = np.hstack([obs, obs_stack])
                exp = np.hstack([exp, context["Expected"][gmpe][imtx]["Mean"]])
                std = np.hstack([std, context["Expected"][gmpe][imtx]["Total"]])
            obs_wrt_imt[imtx] = obs
            exp_wrt_imt[imtx] = exp
            std_wrt_imt[imtx] = std

        return obs_wrt_imt, exp_wrt_imt, std_wrt_imt
    
    def _compute_edr(self, obs, exp, std, bandwidth=0.01, multiplier=3.0):
        """
        Calculate the Euclidean Distanced-Based Rank for a set of
        observed and expected values from a particular GMPE
        """
        finite = np.isfinite(obs) & np.isfinite(exp) & np.isfinite(std)
        if not finite.any():
            return np.nan, np.nan, np.nan
        elif not finite.all():
            obs, exp, std = obs[finite], exp[finite], std[finite]
        nvals = len(obs)
        min_d = bandwidth / 2.
        kappa = self._get_edr_kappa(obs, exp)
        mu_d = obs - exp
        d1c = np.fabs(obs - (exp - (multiplier * std)))
        d2c = np.fabs(obs - (exp + (multiplier * std)))
        dc_max = ceil(np.max(np.array([np.max(d1c), np.max(d2c)])))
        num_d = len(np.arange(min_d, dc_max, bandwidth))
        mde = np.zeros(nvals)
        for iloc in range(0, num_d):
            d_val = (min_d + (float(iloc) * bandwidth)) * np.ones(nvals)
            d_1 = d_val - min_d
            d_2 = d_val + min_d
            p_1 = norm.cdf((d_1 - mu_d) / std) - norm.cdf(
                (-d_1 - mu_d) / std)
            p_2 = norm.cdf((d_2 - mu_d) / std) - norm.cdf(
                (-d_2 - mu_d) / std)
            mde += (p_2 - p_1) * d_val
        inv_n = 1.0 / float(nvals)
        mde_norm = np.sqrt(inv_n * np.sum(mde ** 2.))
        edr = np.sqrt(kappa * inv_n * np.sum(mde ** 2.))

        return mde_norm, np.sqrt(kappa), edr            
    
    def _compute_edr_wrt_imt(self,
                             obs_wrt_imt,
                             exp_wrt_imt,
                             std_wrt_imt,
                             bandwidth=0.01,
                             multiplier=3.0):
        """
        Calculate the Euclidean Distanced-Based Rank for a set of
        observed and expected values from a particular GMPE over IMTs
        """
        mde_norm_wrt_imt = {}
        edr_wrt_imt = {}
        kappa_wrt_imt = {}

        for imtx in self.imts:
            nvals = len(obs_wrt_imt[imtx])
            min_d = bandwidth / 2.
            kappa_wrt_imt[imtx] = self._get_edr_kappa(obs_wrt_imt[imtx],
                                                      exp_wrt_imt[imtx])
            mu_d = obs_wrt_imt[imtx] - exp_wrt_imt[imtx]
            d1c = np.fabs(obs_wrt_imt[imtx] - (exp_wrt_imt[imtx] - (
                multiplier * std_wrt_imt[imtx])))
            d2c = np.fabs(obs_wrt_imt[imtx] - (exp_wrt_imt[imtx] + (
                multiplier * std_wrt_imt[imtx])))
            dc_max = ceil(np.max(np.array([np.max(d1c), np.max(d2c)])))
            num_d = len(np.arange(min_d, dc_max, bandwidth))
            mde_wrt_imt = np.zeros(nvals)
            for iloc in range(0, num_d):
                d_val = (min_d + (float(iloc) * bandwidth)) * np.ones(nvals)
                d_1 = d_val - min_d
                d_2 = d_val + min_d
                p_1 = norm.cdf((d_1 - mu_d) / std_wrt_imt[imtx]) -\
                norm.cdf((-d_1 - mu_d) / std_wrt_imt[imtx])
                p_2 = norm.cdf((d_2 - mu_d) / std_wrt_imt[imtx]) -\
                norm.cdf((-d_2 - mu_d) / std_wrt_imt[imtx])
                mde_wrt_imt += (p_2 - p_1) * d_val
            inv_n = 1.0 / float(nvals)
            mde_norm_wrt_imt[imtx] = np.sqrt(inv_n * np.sum(mde_wrt_imt ** 2.))
            edr_wrt_imt[imtx] = np.sqrt(
                kappa_wrt_imt[imtx] * inv_n * np.sum(mde_wrt_imt ** 2.))

        return mde_norm_wrt_imt, np.sqrt(pd.Series(kappa_wrt_imt)), edr_wrt_imt            

    def _get_edr_kappa(self, obs, exp):
        """
        Returns the correction factor kappa
        """
        mu_a = np.mean(obs)
        mu_y = np.mean(exp)
        b_1 = np.sum(
            (obs - mu_a) * (exp - mu_y)) / np.sum((obs - mu_a) ** 2.)
        b_0 = mu_y - b_1 * mu_a
        y_c = exp - ((b_0 + b_1 * obs) - obs)
        de_orig = np.sum((obs - exp) ** 2.)
        de_corr = np.sum((obs - y_c) ** 2.)

        return de_orig / de_corr

    ### Stochastic Area (Sunny et al. 2021) functions
    def get_stochastic_area_wrt_imt(self):
        """
        Calculates the stochastic area values per imt for each GMPE according
        to the Stochastic Area Ranking method of Sunny et al. (2021).
        
        Sunny, J., M. DeAngelis, and B. Edwards (2021). Ranking and Selection
        of Earthquake Ground Motion Models Using the Stochastic Area Metric,
        Seismol. Res. Lett. 93, 787–797, doi: 10.1785/0220210216
        """
        # Create store of values per gmm
        stoch_area_store = {gmpe: {} for gmpe in self.gmpe_list}
        
        # Get the observed and predicted per gmm per imt
        for gmpe in self.gmpe_list:
            stoch_area_wrt_imt = {}
            for imtx in self.imts:
                obs = np.array([], dtype=float)
                exp = np.array([], dtype=float)
                std = np.array([], dtype=float)
                for context in self.contexts:
                    obs = np.hstack([obs, np.log(context["Observations"][imtx])])
                    exp = np.hstack([exp, context["Expected"][gmpe][imtx]["Mean"]])
                    std = np.hstack([std, context["Expected"][gmpe][imtx]["Total"]])
                
                # Get the ECDF for distribution from observations
                x_ecdf, y_ecdf = self.get_cdf_data(list(obs), step_flag=True)
                
                # Get the CDF for distribution from gmm
                x_cdf, y_cdf = self.get_cdf_data(list(exp))

                # Get approximately overlapping subsets of ECDF and CDF
                ecdf_xvals = [np.nanmin(x_ecdf), np.nanmax(x_ecdf)]
                cdf_xvals = [np.nanmin(x_cdf), np.nanmax(x_cdf)]
                xval_min = np.max([ecdf_xvals[0], cdf_xvals[0]])
                xval_max = np.min([ecdf_xvals[1], cdf_xvals[1]])
                idx_ecdf = np.logical_and(x_ecdf<=xval_max, x_ecdf>=xval_min)
                idx_cdf = np.logical_and(x_cdf<=xval_max, x_cdf>=xval_min)
                x_ecdf, y_ecdf = x_ecdf[idx_ecdf], y_ecdf[idx_ecdf]
                x_cdf, y_cdf = x_cdf[idx_cdf], y_cdf[idx_cdf]

                # Get area under each curve's overlapping portions
                area_obs = trapezoid(y_ecdf, x_ecdf)
                area_gmm = trapezoid(y_cdf, x_cdf)

                # Get absolute of difference in areas - eq 3 of paper
                stoch_area_wrt_imt[imtx] = np.abs(area_gmm-area_obs) 
             
            # Store the stoch area per imt per gmm
            stoch_area_store[gmpe] = stoch_area_wrt_imt
    
        # Add to residuals object
        self.stoch_areas_wrt_imt = stoch_area_store

        return self.stoch_areas_wrt_imt

    def cdf(self, data):
        """
        Get the cumulative distribution function (cdf) of the ground-motion
        values
        """
        x1 = np.sort(data)
        x = x1.tolist()
        n = len(x)
        p = 1/n
        pvalues = list(np.linspace(p,1,n))
        
        return x, pvalues
    
    def step_data(self, x,y):
        """
        Step the cdf to obtain the ecdf
        """
        xx, yy = x*2, y*2
        xx.sort()
        yy.sort()
        return xx, [0.]+yy[:-1]
    
    def get_cdf_data(self, data, step_flag=False):
        """
        Get the cdf (for the predicted ground-motions) or the ecdf (for the
        observed ground-motions)
        """
        x, p = self.cdf(data)
        if step_flag is True:
            xx, yy = self.step_data(x, p)
            return np.array(xx), np.array(yy)
        else:
            return np.array(x), np.array(p)
    

class SingleStationAnalysis(object):
    """
    Residuals object for single station residual analysis.
    """
    def __init__(self, site_id_list, gmpe_list, imts):
        # station sites are strings like 'MN-PDG', 'HL-KASA', ...
        # we sort them lexicographically since the order they are
        # stored in the database is unspecified
        self.site_ids = site_id_list
        if len(self.site_ids) < 1:
            raise ValueError('No sites meet record threshold for analysis.')
        # Copy the GMMs to avoid recursive issues with check_gsim_list 
        self.frozen_gmpe_list = copy.deepcopy(gmpe_list) 
        self.gmpe_list = check_gsim_list(gmpe_list)
        self.imts = imts
        self.site_residuals = []
        self.types = {gmpe: {} for gmpe in self.gmpe_list}
        for gmpe in self.gmpe_list:
            gmpe_i = self.gmpe_list[gmpe]
            for imtx in self.imts:
                self.types[gmpe][imtx] = []
                if gmpe_i.DEFINED_FOR_STANDARD_DEVIATION_TYPES == ALL_SIGMA:
                    for res_type in ['Total','Inter event', 'Intra event']:
                        self.types[gmpe][imtx].append(res_type)
                else:
                    for res_type in (
                        gmpe_i.DEFINED_FOR_STANDARD_DEVIATION_TYPES):
                        self.types[gmpe][imtx].append(res_type)
                        
    @classmethod
    def from_toml(cls, site_id_list, filename):
        """
        Read in GMPEs and IMTs from .toml file. This method allows use of
        gmpes with additional parameters and input files within the SMT
        """
        # Read in toml file with dict of GMPEs and subdict of IMTs
        config = toml.load(filename)
             
        # Parsing file with models
        gmpe_list = []
        for _, key in enumerate(config['models']):
            
            # Get toml representation of GMM
            gmm = get_gmm_from_toml(key, config)

            # Create valid gsim object
            gmpe_list.append(gmm)
            
        # Get imts    
        imts = config['imts']['imt_list']  

        return cls(site_id_list, gmpe_list, imts)

    def get_site_residuals(self, database, component="Geometric"):
        """
        Calculates the total, inter-event and within-event residuals for
        each site
        """
        for site_id in self.site_ids:
            selector = SMRecordSelector(database)
            site_db = selector.select_from_site_id(site_id, as_db=True)
            # Use a deep copied gmpe list to avoid recursive GMM instantiation
            # issues when using check_gsim_list within Residuals obj __init__
            resid = Residuals(self.frozen_gmpe_list, self.imts)
            resid.compute_residuals(site_db, normalise=False, component=component)
            setattr(
                resid,
                "site_analysis",
                self._set_empty_dict())
            setattr(
                resid,
                "site_expected",
                self._set_empty_dict())
            self.site_residuals.append(resid)

    def _set_empty_dict(self):
        """
        Sets an empty set of nested dictionaries for each GMPE and each IMT
        """
        return {gmpe: {imtx: {} for imtx in self.imts} for gmpe in self.gmpe_list}

    def station_residual_statistics(self, filename=None):
        """
        Get single-station residual statistics for each site
        """
        output_resid = []
        for t_resid in self.site_residuals:
            resid = copy.deepcopy(t_resid)
            for gmpe in self.gmpe_list:
                for imtx in self.imts:

                    # If residuals for given GMM-IMT combination
                    if not t_resid.residuals[gmpe][imtx]:
                        continue
                    
                    # Get number events, total residuals, total (GMM) expected
                    n_events = len(t_resid.residuals[gmpe][imtx]["Total"])
                    total_res = np.copy(t_resid.residuals[gmpe][imtx]["Total"])
                    total_exp = np.copy(t_resid.modelled[gmpe][imtx]["Total"])

                    # Store
                    resid.site_analysis[gmpe][imtx]["events"] = n_events
                    resid.site_analysis[gmpe][imtx]["Total"] = total_res
                    resid.site_analysis[gmpe][imtx]["Expected total"] = total_exp
                    
                    if not "Intra event" in t_resid.residuals[gmpe][imtx]:
                        # GMPE has no within-event term - skip
                        continue

                    # Get deep copy of phi (intra) and tau (inter)
                    resid.site_analysis[gmpe][imtx]["Intra event"] = np.copy(
                        t_resid.residuals[gmpe][imtx]["Intra event"])
                    resid.site_analysis[gmpe][imtx]["Inter event"] = np.copy(
                        t_resid.residuals[gmpe][imtx]["Inter event"])

                    # Get delta_s2ss
                    delta_s2ss = self._get_delta_s2ss(
                        resid.residuals[gmpe][imtx]["Intra event"], n_events)
                    
                    # Get delta_woes
                    delta_woes = (
                        resid.site_analysis[gmpe][imtx]["Intra event"] - delta_s2ss)
                    
                    # Get phi_ss
                    phi_ss = self._get_single_station_phi(
                        resid.residuals[gmpe][imtx]["Intra event"], delta_s2ss, n_events)

                    # Store 
                    resid.site_analysis[gmpe][imtx]["dS2ss"] = delta_s2ss
                    resid.site_analysis[gmpe][imtx]["dWo,es"] = delta_woes
                    resid.site_analysis[gmpe][imtx]["phi_ss,s"] = phi_ss
                    
                    # Get expected values too
                    resid.site_analysis[gmpe][imtx]["Expected inter"] =\
                        np.copy(t_resid.modelled[gmpe][imtx]["Inter event"])
                    resid.site_analysis[gmpe][imtx]["Expected intra"] =\
                        np.copy(t_resid.modelled[gmpe][imtx]["Intra event"])
            
            # Store
            output_resid.append(resid)
        
        # Update
        self.site_residuals = output_resid

        return self.get_total_phi_ss(filename)

    def _get_delta_s2ss(self, intra_event, n_events):
        """
        Returns the average within-event residual for the site from
        Rodriguez-Marek et al. (2011) Equation 8
        """
        return (1. / float(n_events)) * np.sum(intra_event)

    def _get_single_station_phi(self, intra_event, delta_s2ss, n_events):
        """
        Returns the single-station phi for the specific station from
        Rodriguez-Marek et al. (2011) Equation 11
        """
        return np.sqrt(
            np.sum((intra_event - delta_s2ss) ** 2.) / float(n_events - 1))

    def get_total_phi_ss(self, filename=None):
        """
        Returns the station averaged single-station phi from Rodriguez-Marek
        et al. (2011) Equation 10
        """
        if filename is not None:
            fid = open(filename, "w")
        else:
            fid = sys.stdout
        phi_ss = self._set_empty_dict()
        phi_s2ss = self._set_empty_dict()
        for gmpe in self.gmpe_list:
            
            # Print GMM info to file
            if filename is not None:
                gmpe_str = get_gmpe_str(gmpe)
                print("%s" % gmpe_str, file=fid)
            
            # Print IMT info to file
            for imtx in self.imts:
                if filename is not None:
                    print("%s" % imtx, file=fid)
                if not ("Intra event" in self.site_residuals[
                    0].site_analysis[gmpe][imtx]):
                    msg = (f"GMPE {gmpe} does not have random"
                           f"effects residuals for {imtx}")
                    warnings.warn(msg, stacklevel=10)
                    continue
                n_events = []
                numerator_sum = 0.0
                d2ss = []
                for iloc, resid in enumerate(self.site_residuals):
                    d2ss.append(resid.site_analysis[gmpe][imtx]["dS2ss"])
                    n_events.append(resid.site_analysis[gmpe][imtx]["events"])
                    numerator_sum += np.sum((
                        resid.site_analysis[gmpe][imtx]["Intra event"] -
                        resid.site_analysis[gmpe][imtx]["dS2ss"]) ** 2.)
                    
                    # Print dS2S, phi_ss per station to file
                    if filename is not None:
                        print("Site ID, %s, dS2S, %s, "
                              "phi_ss, %s, Num Records, %s" % (
                              list(self.site_ids)[iloc],
                              resid.site_analysis[gmpe][imtx]["dS2ss"],
                              resid.site_analysis[gmpe][imtx]["phi_ss,s"],
                              resid.site_analysis[gmpe][imtx]["events"]),
                              file=fid)
                        
                d2ss = np.array(d2ss)
                phi_s2ss[gmpe][imtx] = {
                    "Mean": np.mean(d2ss), "StdDev": np.std(d2ss)}
                phi_ss[gmpe][imtx] = np.sqrt(
                    numerator_sum / float(np.sum(np.array(n_events)) - 1))
        
        # Print phi_ss (single-station phi), phi_s2s (station-to-station) to file
        if filename is not None:
            print("\nSSA RESULTS PER GMPE", file=fid)
            for gmpe in self.gmpe_list:
                gmpe_i = self.gmpe_list[gmpe]
                gmpe_str = get_gmpe_str(gmpe)
                print("%s" % gmpe_str, file=fid)
                if gmpe_i.DEFINED_FOR_STANDARD_DEVIATION_TYPES == ALL_SIGMA:
                    p_data = (imtx,
                              phi_ss[gmpe][imtx],
                              phi_s2ss[gmpe][imtx]["Mean"],
                              phi_s2ss[gmpe][imtx]["StdDev"])
                else:
                    p_data = (imtx, None, None, None) # No intra-event for GMM
                for imtx in self.imts:                # so write blank values
                    print("%s, "\
                          "phi_ss (phi single-station), %s" \
                          "phi_s2s mean, %s, " \
                          "phi_s2s std. dev, %s" \
                          % p_data, file=fid)
                            
            if filename is not None:
                fid.close()

        return phi_ss, phi_s2ss
