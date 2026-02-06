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
Module to get GMPE residuals.
"""
import sys
import warnings
import copy
import re
import toml
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.stats import norm

from openquake.hazardlib import imt, valid, nrml, contexts
from openquake.baselib.node import Node as N
from openquake.hazardlib.gsim_lt import GsimLogicTree

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
          "ry0",
          "rvolc",
          "rcdpp"]


### Util functions
def get_gmm_from_toml(key, config):
    """
    Get a GMM from a TOML file.
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

    # Get GMM
    gmm = valid.gsim(value.strip())

    # HACK: Also make sure still retrieving any rup, dist and site
    # params only specified in the parent class (sometimes the use
    # of gsim aliases means they are not added as expected)
    parent = gmm.__class__.__bases__[0]
    if parent.__name__ != "GMPE": # Must be a subclass
        # Rup params
        for par in parent.REQUIRES_RUPTURE_PARAMETERS:
            if par not in gmm.REQUIRES_RUPTURE_PARAMETERS:
                gmm.REQUIRES_RUPTURE_PARAMETERS |= {par}
        # Site params
        for par in parent.REQUIRES_SITES_PARAMETERS:
            if par not in gmm.REQUIRES_SITES_PARAMETERS:
                gmm.REQUIRES_SITES_PARAMETERS |= {par}
        # Dist params
        for par in parent.REQUIRES_DISTANCES:
            if par not in gmm.REQUIRES_DISTANCES:
                gmm.REQUIRES_DISTANCES |= {par}

    return gmm


def get_gmpe_str(gmpe):
    """
    Return a simplified string representative of the given gmpe.
    """
    if '_toml=' in str(gmpe):
        return str(gmpe).split('_toml=')[1].replace(')','').replace('\n','; ')
    else:
        return gmpe


def get_mean_stds(rup_ctx, gsim, imt):
    """
    :param rup_ctx: a RuptureContext with site information
    :param gsim: a GSIM instance
    :param imt_str: an IMT string
    :return: 4 arrays (mean, sig, tau, phi) of N elements each.
    """
    cmaker = contexts.simple_cmaker([gsim], [imt])
    ctx = cmaker.recarray([rup_ctx])
    return cmaker.get_mean_stds([ctx])[:, 0, 0, :]  # (4, N)


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

        sa = any("SA(" in imtx for imtx in self.imts)
        for gmpe in self.gmpe_list:
            gmpe_dict_1 = {}
            gmpe_dict_2 = {}
            self.unique_indices[gmpe] = {}
            
            # If evaluting GMMs for SA then get the min/max periods
            gmpe_i = self.gmpe_list[gmpe]
            coeff_atts = [att for att in dir(gmpe_i) if "COEFFS" in att]
            if len(coeff_atts) > 0:
                coeff_att = coeff_atts[0] # Some GSIMS have irreg. COEFF attribute 
                                          # names e.g. Z06 (but const. period range)
                pers = [sa.period for sa in getattr(gmpe_i, coeff_att).sa_coeffs]
                if len(pers) == 0 and sa is True:
                    raise ValueError(f"No period-dependent coefficients could be "
                                     f"retrieved for {get_gmpe_str(gmpe)} - check "
                                     f"that this GMM supports SA.")
                self.gmpe_scalars[gmpe] = list(
                    getattr(gmpe_i, coeff_att).non_sa_coeffs)
            else:
                assert hasattr(gmpe_i, "gmpe_table")
                # Tabular GMM specified using an alias
                pers = gmpe_i.imls["T"]

            # Store min/max periods for given GMM
            if sa is True:
                min_per, max_per = (min(pers), max(pers))
                self.gmpe_sa_limits[gmpe] = (min_per, max_per)

            # Add stores for each IMT
            for imtx in self.imts:
                if "SA(" in imtx:
                    period = imt.from_string(imtx).period
                    if period < min_per or period > max_per:
                        raise ValueError(
                            f"IMT {imtx} outside period range for {gmpe} "
                            f"(min GMM period = {min_per} s, "
                            f"max GMM period = {max_per} s)")
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
        Read in gmpe_list and imts from .toml file.
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

    @classmethod
    def from_xml(cls, filename, imts):
        """
        Read in the GMMs from an XML and the IMTs as list of IMTs.

        NOTE: We read all of the GMMs over the potentially multiple
        branchsets. If the user wishes to evaluate only one branchset
        (i.e. for one TRT, which is more likely), then they should just
        remove the not-required branchsets from the XML.
        """
        # Get the GMMs from the xml
        gmpe_list = [gmm.gsim for gmm in GsimLogicTree(filename).branches]

        return cls(gmpe_list, imts)

    def compute_residuals(self,
                          ctx_database,
                          nodal_plane_index=1,
                          component="Geometric",
                          normalise=True,
                          stations=False):
        """
        Calculate the residuals for a set of ground motion records

        :param ctx_database: a :class:`context_db.ContextDB`, i.e. a database of
            records capable of returning dicts of earthquake-based Contexts and
            observed IMTs.
            See e.g., :class:`openquake.smt.sm_database.GroundMotionDatabase`
            for an example
        :param stations: Bool which if set to True prevents an error being raised
                         if all obs values for given IMT are nans at the station,
                         which is forbidden for a single ground-motion record in
                         a regular residual analysis, but permitted when computing
                         single-station residuals
        """
        # Build initial contexts with the observed values
        contexts = ctx_database.get_contexts(nodal_plane_index,
                                             self.imts,
                                             component)
        
        # Check at least one observed value per IMT (else raise an error)
        for im in self.imts:
            obs_check = []
            for ctx in contexts:
                obs = ctx["Observations"][im]
                if stations is True:
                    # In SSA should be one rec per ev
                    # given computing res per station
                    assert len(obs) == 1
                obs_check.append(obs)
            obs_check = np.concatenate(obs_check)
            check = pd.notnull(obs_check)
            if len(check[check]) < 1 and stations is False:
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
            # If units are acceleration (admitted in cm/s/s) to g
            for a_imt in accel_imts:
                context['Observations'][
                    a_imt] = convert_accel_units(
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

        for gmpe in self.residuals:
            for imtx in self.residuals[gmpe]:
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
        Calculate the expected ground motions from the context.
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
                mean, *stddev = get_mean_stds(context["Ctx"], gsim, imtx)
                keep = context["Retained"][imtx]
                mean = mean[keep]
                for idx_comp, comp in enumerate(stddev):
                    stddev[idx_comp] = comp[keep]
                # If no sigma for the GMM residuals can't be computed
                if np.all(stddev[0] == 0.) and len(keep) > 0:
                    gs = str(gmpe).split('(')[0]
                    mg = 'A sigma model is not provided for %s' %gs
                    raise ValueError(mg)
                exp[gmpe][imtx]["Mean"] = mean
                for i, res_type in enumerate(self.types[gmpe][imtx]):
                    exp[gmpe][imtx][res_type] = stddev[i]

        context["Expected"] = exp

        return context

    def calculate_residuals(self, context, normalise=True):
        """
        Calculate the residual terms.
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
                        normalise
                        )
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
        Calculates the random effects residuals (i.e. decomposition of the
        total residuals into inter-event and intra-event) using equation 10
        of Abrahamson & Youngs (1992).
        
        :param obs: array of observed ground-shaking values for a single ctx
                    (i.e. event) for a given imt, in natural log
        :param mean: array of ground-shaking values for the same ctx 
                     predicted by the given GMPE and imt, in natural log
        :param inter: float representing the inter-event component of GMPE
                      sigma for a given imt
        :param intra: float representing the intra-event component of GMPE
                      sigma for a given imt
        :param normalise: bool which if True normalises the residuals using
                          the corresponding GMPE sigma components
        """
        # Get number of values
        nvals = float(len(mean))

        # Total variance for all observations combining GMPE tau and phi
        v = nvals * (inter ** 2.) + (intra ** 2.)
                                                  
        # Compute the inter-event
        inter_res = ((inter ** 2.) * sum(obs - mean)) / v

        # Compute the intra-event
        intra_res = obs - (mean + inter_res)

        # Whether to normalise or not
        if normalise:
            return inter_res / inter, intra_res / intra
        else:
            return inter_res, intra_res

    def get_residual_statistics(self):
        """
        Retreives the mean and standard deviation values of the residuals.
        """
        statistics = {gmpe: {} for gmpe in self.gmpe_list}
        for gmpe in self.gmpe_list:
            for imtx in self.imts:
                if not self.residuals[gmpe][imtx]:
                    continue
                statistics[gmpe][imtx] = self.get_residual_statistics_for(gmpe, imtx)
                
        return statistics

    def get_residual_statistics_for(self, gmpe, imtx):
        """
        Retreives the mean and standard deviation values of the residuals for
        a given gmpe and imtx.
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
        residuals.
        """
        magnitudes = np.array([])
        for ctxt in self.contexts:
            magnitudes = np.hstack([
                magnitudes,
                ctxt["Ctx"].mag * np.ones(len(ctxt["Ctx"].repi))])
            
        return magnitudes

    def export_residuals(self, out_fname):
        """
        Export the observed, predicted and residuals to a text file.
        """
        ctxs = self.contexts # List of contexts
        store = {}
        for ctx in ctxs:
            for imt in self.imts:
                ctx_and_imt = {} # One df per imt and ctx
                for gmpe in self.gmpe_list:                
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

    def export_gmc_xml(self, weight_metric, out_fname):
        """
        Export the GMMs evaluated in the residual analysis to an OQ GMC XML.
        The weights of each GMM can be based on the normalisation of the LLH,
        EDR or Stochastic Area scores (averaged over all considered IMTs).

        NOTE: This function sets a default TRT of "*". Once written to XML
        the user must modify this to match the TRT they wish to apply the
        exported logic tree to within their seismic source model.

        :param weight_metric: Can be "LLH", "EDR", "STO" or "equal".
        """
        # Map the scores to attributes in residuals object
        score_map = {"LLH": "llh_weights",
                     "EDR": "edr_weights",
                     "STO": "sto_weights",
                     "equal": None}

        # Check weight metric is valid
        if weight_metric not in score_map.keys():
            raise ValueError(f"An invalid weight metric has been"
                             f"specified for GMC XML exporting - "
                             f"must be in {list(score_map.keys())}")
        
        # Check required weights are in residuals obj
        if weight_metric != "equal":
            if not hasattr(self, score_map[weight_metric]):
                raise ValueError(
                    f"Cannot use {weight_metric} weights because "
                    f"{score_map[weight_metric]} attribute is missing "
                    f"from residuals obj (you must first compute the "
                    f"{weight_metric}-based weights).")

        # Get the weights
        if weight_metric != "equal":
            weights = getattr(self, score_map[weight_metric])
        else:
            weights = {gmm: 1/len(self.gmpe_list) for gmm in self.gmpe_list}
        
        # Make a branch for each GMM
        branches = []
        for idx_gmm, gmm in enumerate(self.gmpe_list):
            if weight_metric != "equal":
                wei = weights[f"{gmm} {weight_metric}-based weight"]['Avg over imts']
            else:
                wei = weights[gmm]
        
            # Make the branch
            branch = N('logicTreeBranch', {'branchID': f'b{idx_gmm}'}, 
                        nodes=[N('uncertaintyModel', text=str(gmm)),
                               N('uncertaintyWeight', text=str(wei))])
            
            # Store
            branches.append(branch)

        # Make an LT
        lt = N('logicTree', {'logicTreeID': 'lt1'},
               nodes=[N('logicTreeBranchSet',
                        {'applyToTectonicRegionType': '*',
                         'branchSetID': 'bs1',
                         'uncertaintyType': 'gmpeModel'},
                         nodes=branches)])
        gsim_lt = GsimLogicTree('<in-memory>', ['*'], ltnode=lt)

        # Write to XML
        with open(out_fname, 'wb') as f:
            nrml.write([gsim_lt.to_node()], f)

    ### LLH (Scherbaum et al. 2009) functions
    def get_llh_values(self):
        """
        Returns the loglikelihood fit of the GMPEs to data using the
        loglikehood (LLH) function described in Scherbaum et al. (2009):
        
        Scherbaum, F., Delavaud, E., Riggelsen, C. (2009) "Model Selection in
        Seismic Hazard Analysis: An Information-Theoretic Perspective",
        Bulletin of the Seismological Society of America, 99(6), 3234-3247
        """
        # Iterate over the GMMs
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
                self.llh[gmpe][imtx] = -1 * (1.0 / float(len(asll))) * np.sum(asll)
            
                # Stack
                log_residuals = np.hstack([log_residuals, asll])

            # Get the average over the IMTs
            self.llh[gmpe]["all"] = -1 * (
                1. / float(len(log_residuals))) * np.sum(log_residuals)

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
        # Set store
        self.edr_values = {gmpe: {} for gmpe in self.gmpe_list}

        # Iterate over the GMMs
        for gmpe in self.gmpe_list:
            
            # Set empty arrays
            obs = np.array([], dtype=float)
            exp = np.array([], dtype=float)
            std = np.array([], dtype=float)

            # Stack over the IMTs
            for imtx in self.imts:
                for context in self.contexts:
                    keep = context["Retained"][imtx]
                    obs = np.hstack([obs, np.log(context["Observations"][imtx][keep])])
                    exp = np.hstack([exp, context["Expected"][gmpe][imtx]["Mean"]])
                    std = np.hstack([std, context["Expected"][gmpe][imtx]["Total"]])

            # Now compute EDR
            results = self._compute_edr(obs, exp, std, bandwidth, multiplier)

            # Store
            self.edr_values[gmpe]["MDE Norm"] = results[0]
            self.edr_values[gmpe]["sqrt Kappa"] = results[1]
            self.edr_values[gmpe]["EDR"] = results[2]

    def get_edr_wrt_imt(self, bandwidth=0.01, multiplier=3.0):
        """
        Calculates the EDR values for each GMPE but per IMT instead.

        :param float bandwidth:
            Discretisation width

        :param float multiplier:
            "Multiplier of standard deviation (equation 8 of Kale and Akkar)
        """
        # Set store
        self.edr_values_wrt_imt = {gmpe: {key: {
            imtx: None for imtx in self.imts} for key in [
                "MDE Norm", "sqrt Kappa", "EDR"]} for gmpe in self.gmpe_list}

        # Iterate over the GMMs
        for gmpe in self.gmpe_list:

            # Iterate over IMTs
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
               
                # Compute EDR for given IMT
                results = self._compute_edr(obs, exp, std, bandwidth, multiplier)
                
                # Store
                self.edr_values_wrt_imt[gmpe]["MDE Norm"][imtx] = results[0]
                self.edr_values_wrt_imt[gmpe]["sqrt Kappa"][imtx]= results[1]
                self.edr_values_wrt_imt[gmpe]["EDR"][imtx] = results[2]
    
    def _compute_edr(self, obs, exp, std, bandwidth=0.01, multiplier=3.0):
        """
        Calculate the Euclidean Distanced-Based Rank for a set of
        observed and expected values from a particular GMPE.
        """
        finite = np.isfinite(obs) & np.isfinite(exp) & np.isfinite(std)
        if not finite.any():
            return np.nan, np.nan, np.nan
        obs, exp, std = obs[finite], exp[finite], std[finite]
        nvals = len(obs)
        min_d = bandwidth / 2.
        kappa = self._get_edr_kappa(obs, exp)
        mu_d = obs - exp
        d1c = np.fabs(obs - (exp - (multiplier * std)))
        d2c = np.fabs(obs - (exp + (multiplier * std)))
        dc_max = np.ceil(np.max(np.array([np.max(d1c), np.max(d2c)])))
        num_d = len(np.arange(min_d, dc_max, bandwidth))
        mde = np.zeros(nvals)
        for iloc in range(0, num_d):
            d_val = (min_d + (float(iloc) * bandwidth)) * np.ones(nvals)
            d_1 = d_val - min_d
            d_2 = d_val + min_d
            p_1 = norm.cdf((d_1 - mu_d) / std) - norm.cdf((-d_1 - mu_d) / std)
            p_2 = norm.cdf((d_2 - mu_d) / std) - norm.cdf((-d_2 - mu_d) / std)
            mde += (p_2 - p_1) * d_val
        inv_n = 1.0 / float(nvals)
        mde_norm = np.sqrt(inv_n * np.sum(mde ** 2.))
        edr = np.sqrt(kappa * inv_n * np.sum(mde ** 2.))

        return mde_norm, np.sqrt(kappa), edr                

    def _get_edr_kappa(self, obs, exp):
        """
        Returns the correction factor kappa.
        """
        mu_a = np.mean(obs)
        mu_y = np.mean(exp)
        b_1 = np.sum((obs - mu_a) * (exp - mu_y)) / np.sum((obs - mu_a) ** 2.)
        b_0 = mu_y - b_1 * mu_a
        y_c = exp - ((b_0 + b_1 * obs) - obs)
        de_orig = np.sum((obs - exp) ** 2.)
        de_corr = np.sum((obs - y_c) ** 2.)
        
        return de_orig / de_corr

    ### Stochastic Area (Sunny et al. 2021) functions
    def get_sto_wrt_imt(self):
        """
        Calculates the stochastic area values per GMPE for each IMT
        according to the Stochastic Area Ranking method of Sunny et
        al. (2021):
        
        Sunny, J., M. DeAngelis, and B. Edwards (2021). Ranking and Selection
        of Earthquake Ground Motion Models Using the Stochastic Area Metric,
        Seismol. Res. Lett. 93, 787–797, doi: 10.1785/0220210216
        """
        # Create store of values per gmm
        stoch_area_store = {gmpe: {} for gmpe in self.gmpe_list}
        
        # Iterate over the GMMs
        for gmpe in self.gmpe_list:
            stoch_area_wrt_imt = {}

            # Iterate over the IMTs
            for imtx in self.imts:

                # Stack values
                obs = np.array([], dtype=float)
                exp = np.array([], dtype=float)
                std = np.array([], dtype=float)
                for context in self.contexts:
                    obs = np.hstack([obs, np.log(context["Observations"][imtx])])
                    exp = np.hstack([exp, context["Expected"][gmpe][imtx]["Mean"]])
                    std = np.hstack([std, context["Expected"][gmpe][imtx]["Total"]])

                # Take only the finite obs
                idx_f = np.isfinite(obs)
                obs = obs[idx_f]
                assert len(obs) == len(exp) == len(std)

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
                stoch_area = np.abs(area_gmm - area_obs)

                # Store the stoch area per imt for given gmm
                stoch_area_wrt_imt[imtx] = max(1E-09, stoch_area)

            # Store for given gmm
            stoch_area_store[gmpe] = stoch_area_wrt_imt
        
        # Add to residuals object
        self.stoch_areas_wrt_imt = stoch_area_store

    def cdf(self, data):
        """
        Get the cumulative distribution function (cdf).
        """
        x1 = np.sort(data)
        x = x1.tolist()
        n = len(x)
        p = 1/n
        pvalues = list(np.linspace(p,1,n))
        
        return x, pvalues
    
    def step_data(self, x,y):
        """
        Step the cdf to obtain the ecdf.
        """
        xx, yy = x*2, y*2
        xx.sort()
        yy.sort()
        return xx, [0.]+yy[:-1]
    
    def get_cdf_data(self, data, step_flag=False):
        """
        Get the cdf (for the predicted ground-motions) or the ecdf (for the
        observed ground-motions).
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
        # Station sites are strings like 'MN-PDG', 'HL-KASA', ...
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
        Read in GMPEs and IMTs from .toml file.
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
        each site.
        """
        for site_id in self.site_ids:
            selector = SMRecordSelector(database)
            site_db = selector.select_from_site_id(site_id, as_db=True)
            # Use a deep copied gmpe list to avoid recursive GMM instantiation
            # issues when using check_gsim_list within Residuals obj's init
            resid = Residuals(self.frozen_gmpe_list, self.imts)
            resid.compute_residuals(site_db,
                                    component=component,
                                    stations=True)
            setattr(
                resid,
                "site_analysis",
                {gmpe: {imtx: {} for imtx in self.imts} for gmpe in self.gmpe_list}
                )
            setattr(
                resid,
                "site_expected",
                {gmpe: {imtx: {} for imtx in self.imts} for gmpe in self.gmpe_list}
                )
            self.site_residuals.append(resid)

    def station_residual_statistics(self, filename=None):
        """
        Get single-station residual statistics for each site.

        Equation numbers throughout this function and those called within refer to
        equations provided within Rodriguez-Marek et al. (2011) for the computation
        of the site-specific components of the intra-event residual.
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

                    # Get deep copy of intra and inter residuals
                    resid.site_analysis[gmpe][imtx]["Intra event"] = np.copy(
                        t_resid.residuals[gmpe][imtx]["Intra event"])
                    resid.site_analysis[gmpe][imtx]["Inter event"] = np.copy(
                        t_resid.residuals[gmpe][imtx]["Inter event"])

                    # Get deltaW_es (i.e. the intra-event residuals)
                    deltaW_es = resid.residuals[gmpe][imtx]["Intra event"]

                    # Get deltaS2S_s (avg within-event for the station - eq 8)
                    # NOTE: the std of deltaS2S_s over the stations is phi_S2S
                    deltaS2S_s = np.sum(deltaW_es)/n_events
                    
                    # Get deltaWS_es (within-site residual - eq 9)
                    deltaWS_es = deltaW_es - deltaS2S_s
                    
                    # Get phi_ss,s for given station (i.e. std of deltaWS_es- eq 11)
                    phi_ss_s = np.sqrt(
                        np.sum((deltaWS_es) ** 2.) / float(n_events - 1)
                        )
                    
                    # Store 
                    resid.site_analysis[gmpe][imtx]["deltaS2S_s"] = deltaS2S_s
                    resid.site_analysis[gmpe][imtx]["deltaWS_es"] = deltaWS_es
                    resid.site_analysis[gmpe][imtx]["phi_ss,s"] = phi_ss_s
                    
                    # Get expected values too
                    resid.site_analysis[gmpe][imtx]["Expected inter"] =\
                        np.copy(t_resid.modelled[gmpe][imtx]["Inter event"])
                    resid.site_analysis[gmpe][imtx]["Expected intra"] =\
                        np.copy(t_resid.modelled[gmpe][imtx]["Intra event"])
            
            # Store
            output_resid.append(resid)
        
        # Update
        self.site_residuals = output_resid

        # Now can get station averaged values of (phi_ss and deltaS2S)
        self._get_station_averaged_values(filename)

    def _get_station_averaged_values(self, filename=None):
        """
        Compute station-averaged standard deviation of deltaS2S_s
        (i.e. phi_ss, rather than phi_ss,s which is per station)
        AND station-averaged phiS2S_s (i.e. phiS2S). 
        """
        fid = open(filename, "w") if filename else sys.stdout
        self.mean_deltaS2S = {
            gmpe: {imtx: {} for imtx in self.imts} for gmpe in self.gmpe_list}
        self.phi_S2S = {
            gmpe: {imtx: {} for imtx in self.imts} for gmpe in self.gmpe_list}
        self.phi_ss = {
            gmpe: {imtx: {} for imtx in self.imts} for gmpe in self.gmpe_list}

        for gmpe in self.gmpe_list:
            if fid is not None and fid is not sys.stdout:
                print(get_gmpe_str(gmpe), file=fid)

            for imtx in self.imts:
                if fid is not None and fid is not sys.stdout:
                    print(imtx, file=fid)

                if "Intra event" not in self.site_residuals[0].site_analysis[gmpe][imtx]:
                    warnings.warn(
                        f"GMPE {gmpe} does not have random effects residuals for {imtx}",
                        stacklevel=10,
                    )
                    continue
                
                # Return mean deltaS2S, stddev of deltaS2S (phi_S2S) and phi_ss
                st_averaged = self._compute_station_averaged_values(gmpe, imtx, fid)
                self.mean_deltaS2S[gmpe][imtx] = st_averaged[0]
                self.phi_S2S[gmpe][imtx] = st_averaged[1]
                self.phi_ss[gmpe][imtx] = st_averaged[2]

        if filename is not None:
            # Print the rest of the results to file
            self._print_ssa_results(fid, self.mean_deltaS2S, self.phi_ss, self.phi_S2S)
            fid.close()

    def _compute_station_averaged_values(self, gmpe, imtx, fid):
        """
        Computes the following: 

            1) Mean deltaS2S_s w.r.t. all the stations
        
            2) Stddev of deltaS2S_s w.r.t. all the stations (phi_S2S)
            
            3) Compute station-averaged single-station standard deviation
               (phi_ss) using equation 10 of Rodriguez-Marek et al. (2011)
               
        NOTE: This function returns phi_ss (station-averaged) which is
        NOT phi_ss,s (per station) - the prior is computed assuming a
        homoskedastic model (see equation 10). The user is referred to
        pp. 1248 of Rodriguez-Marek et al. (2011) for more info.
        """
        # Set some stores
        deltaS2S_s, n_events = [], []

        # For each station collect deltaS2S_s and the num. events associated
        numerator_sum = 0.0
        for iloc, resid in enumerate(self.site_residuals):
            site_data = resid.site_analysis[gmpe][imtx]
            deltaS2S_s.append(site_data["deltaS2S_s"])
            n_events.append(site_data["events"])
            numerator_sum += np.sum(
                (site_data["Intra event"] - site_data["deltaS2S_s"]) ** 2)
            if fid is not None and fid is not sys.stdout:
                print(
                    f"Site ID, {list(self.site_ids)[iloc]}, "
                    f"deltaS2S_s, {site_data['deltaS2S_s']}, "
                    f"phi_ss,s, {site_data['phi_ss,s']}, "
                    f"Num Records, {site_data['events']}",
                    file=fid
                )

        # Compute mean deltaS2S_s
        mean_deltaS2S = np.mean(deltaS2S_s)

        # Compute phi_S2S (stddev of deltaS2S_s amongst the stations)
        phi_S2S = np.std(deltaS2S_s)

        # Compute station averaged phi_ss,s (eq 10) for given gmpe and imt
        phi_ss = np.sqrt(numerator_sum / (np.sum(n_events) - 1))

        return mean_deltaS2S, phi_S2S, phi_ss

    def _print_ssa_results(self, fid, mean_deltaS2S, phi_ss, phi_S2S):
        """
        Print SSA results to the file.
        """
        ni = 'Sigma model of GMPE has no intra-event component'
        if fid is not None and fid is not sys.stdout:
            print("\nSSA RESULTS PER GMPE", file=fid)
        for gmpe in self.gmpe_list:
            gmm_str = get_gmpe_str(gmpe)
            gmm_sigmas = valid.gsim(gmm_str).DEFINED_FOR_STANDARD_DEVIATION_TYPES
            if fid is not None and fid is not sys.stdout:
                print(gmm_str, file=fid)
            for imtx in self.imts:
                p_data = (
                    imtx,
                    phi_ss[gmpe][imtx],
                    mean_deltaS2S[gmpe][imtx],
                    phi_S2S[gmpe][imtx],
                ) if gmm_sigmas == ALL_SIGMA else (imtx, ni, ni, ni)
                if fid is not None and fid is not sys.stdout:
                    print(
                        f"{p_data[0]}, "
                        f"phi_ss, {p_data[1]}, "
                        f"deltaS2S, {p_data[2]}, "
                        f"phi_S2S, {p_data[3]}",
                        file=fid
                        )
