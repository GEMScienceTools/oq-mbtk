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
Module defining the interface of a Context Database (ContextDB), a database of
data capable of yielding Contexts and Observations suitable for Residual analysis
"""
import numpy as np
import pandas as pd

from openquake.hazardlib.contexts import DistancesContext, RuptureContext


class ContextDB:
    """
    This abstract-like class represents a database (DB) of data capable of
    yielding Contexts and Observations suitable for residual analysis (see
    argument `ctx_database` of :meth:`gmpe_residuals.Residuals.compute_residuals`).

    NOTE: The missing distance metrics from a record within the database object
    are computed by OQ using the constructed finite rupture (i.e. the distances
    in the admitted flatfile/ground-motion database are used by default).

    Concrete subclasses of `ContextDB` must implement three abstract methods
    (e.g. :class:`openquake.smt.sm_database.GroundMotionDatabase`):
     - get_event_and_records(self)
     - update_context(self, ctx, records, nodal_plane_index=1)
     - get_observations(self, imtx, records, component="Geometric")
       (which is called only if `imts` is given in :meth:`self.get_contexts`)
    """
    rupture_context_attrs = tuple(RuptureContext._slots_)
    distances_context_attrs = tuple(DistancesContext._slots_)
    sites_context_attrs = ('custom_site_id',
                           'vs30',
                           'lons',
                           'lats',
                           'depths',
                           'vs30measured',
                           'z1pt0',
                           'z2pt5',
                           'backarc')

    def get_contexts(self, nodal_plane_index=1, imts=None, component="Geometric"):
        """
        Return an iterable of Contexts. Each Context is a `dict` with
        earthquake, sites and distances information (`dict["Ctx"]`)
        and optionally arrays of observed IMT values (`dict["Observations"]`).
        See `create_context` for details.

        This is the only method required by
        :meth:`gmpe_residuals.Residuals.compute_residuals`
        and should not be overwritten only in very specific circumstances.
        """
        compute_observations = imts is not None and len(imts)
        ctxs = []
        for evt_id, records in self.get_event_and_records():
            dic = self.create_context(evt_id, imts)
            ctx = dic['Ctx']
            self.update_context(ctx, records, nodal_plane_index)
            if compute_observations:
                for imtx, values in dic["Observations"].items():
                    values = self.get_observations(imtx, records, component)
                    check = pd.notnull(values)
                    dic["Observations"][imtx] = np.asarray(values, dtype=float)
                    dic["Retained"][imtx] = np.argwhere(check==True).flatten()
                dic["Num. Sites"] = len(records)
            dic['Ctx'].sids = np.arange(len(records), dtype=np.uint32)
            dic['Ctx'].custom_site_id = [sid.site.id for sid in records]
            ctxs.append(dic)
        return ctxs

    def create_context(self, evt_id, imts=None):
        """
        Create a new Context `dict`. Objects of this type will be yielded
        by `get_context`.

        :param evt_id: the earthquake id (e.g. int, or str)
        :param imts: a list of strings denoting the IMTs to be included in the
            context. If missing or None, the returned dict **will NOT** have
            the keys "Observations" and "Num. Sites"

        :return: the dict with keys:
            ```
            {
            'EventID': evt_id,
            'Ctx: a new :class:`openquake.hazardlib.contexts.RuptureContext`
            'Observations": dict[str, list] # (each imt in imts mapped to `[]`)
            'Num. Sites': 0
            }
            ```
            NOTE: Remember 'Observations' and 'Num. Sites' are missing if `imts`
            is missing, None or an empty sequence.
        """
        dic = {'EventID': evt_id, 'Ctx': RuptureContext()}
        if imts is not None and len(imts):
            dic["Observations"] = {imt: [] for imt in imts}
            dic["Retained"] = {imt: None for imt in imts}
            dic["Num. Sites"] = 0
        return dic
