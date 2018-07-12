import code
import re
import numpy

from openquake.hazardlib.geo.geodetic import distance
from openquake.hazardlib.geo.geodetic import npoints_towards
from openquake.hazardlib.geo.line import Line
from openquake.hazardlib.geo.point import Point
from openquake.hazardlib.geo.geodetic import (min_distance_to_segment,
                                              point_at, azimuth)
from pyproj import Geod

from openquake.hmtk.seismicity.selector import CatalogueSelector
from openquake.hmtk.parsers.catalogue.gcmt_ndk_parser import ParseNDKtoGCMT


class CrossSectionData:
    """
    """

    def __init__(self, cross_section):
        self.csec = cross_section
        self.slab1pt0 = None
        self.ecat = None
        self.trench = None
        self.moho = None
        self.gcmt = None
        self.topo = None
        self.litho = None
        self.volc = None

    def set_trench_axis(self, filename):
        """
        :parameter filename:
            The name of the xy file containing the trench axis
        """
        print('setting trench axis')
        fin = open(filename, 'r')
        trench = []
        for line in fin:
            aa = re.split('\s+', re.sub('^\s+', '', line))
            trench.append((float(aa[0]), float(aa[1])))
        fin.close()
        self.trench = numpy.array(trench)

    def set_catalogue(self, catalogue, bffer=75.):
        """
        """
        print('setting catalogue')
        na,na,na,na,qual = self.csec.get_mm()

        if qual==1:
            idxs = self.csec.get_eqks_within_buffer_idl(catalogue, bffer)
        else:
            idxs = self.csec.get_eqks_within_buffer(catalogue, bffer)
        boo = numpy.zeros_like(catalogue.data['magnitude'], dtype=int)
        boo[idxs] = 1
        selector = CatalogueSelector(catalogue, create_copy=True)
        newcat = selector.select_catalogue(boo)
        self.ecat = newcat

    def set_slab1pt0(self, filename, bffer=2.0):
        """
        :parameter filename:
            The name of a .xyz grid containing Slab 1.0 data
        :parameter bffer:
            Buffer distance [km]
        :return:
            An array
        """
        print('setting slab')
        # Read the Slab 1.0 file
        slab1pt0 = []
        for line in open(filename):
            aa = re.split('\s+', line)
            if not re.search('[a-z]', aa[2]):
                slab1pt0.append([float(aa[0]), float(aa[1]), float(aa[2])])
        slab1pt0or = numpy.asarray(slab1pt0)
        # Get min and max longitude and latitude values
        minlo, maxlo, minla, maxla, qual = self.csec.get_mm()
        # Find the nodes of the grid within a certain distance from the plane
        # of the cross-section
        slab1pt0 = slab1pt0or
        idx = numpy.nonzero(slab1pt0or[:,0] > 180)
        if len(idx[0]):
            slab1pt0[idx[0], 0] = slab1pt0[idx[0], 0] - 360.
        if qual==0:
            minlo, maxlo, minla, maxla, qual = self.csec.get_mm(2.0)
            idxslb = self.csec.get_grd_nodes_within_buffer(slab1pt0[:,0],
                                                           slab1pt0[:,1],
                                                           bffer,
                                                           minlo, maxlo,
                                                           minla, maxla)
        if qual==1:
           idxslb = self.csec.get_grd_nodes_within_buffer_idl(slab1pt0[:,0],
                                                            slab1pt0[:,1],
                                                            bffer,
                                                            minlo, maxlo,
                                                            minla, maxla)
        if idxslb is not None:
            self.slab1pt0 = numpy.squeeze(slab1pt0[idxslb, :])


    def set_crust1pt0_moho_depth(self, filename, bffer=100.):
        """
        :parameter filename:
            The name of the file containing the CRUST 1.0 model
        :return:
            A numpy array
        """
        print('setting crust/moho')
        datal = []
        for line in open(filename, 'r'):
            xx = re.split('\s+', re.sub('\s+$', '', re.sub('^\s+', '', line)))
            datal.append([float(val) for val in xx])
        dataa = numpy.array(datal)
        minlo, maxlo, minla, maxla, qual = self.csec.get_mm()
        if qual==0:
            minlo, maxlo, minla, maxla, qual = self.csec.get_mm(2.0)
            idxs = self.csec.get_grd_nodes_within_buffer(dataa[:,0],
                                                     dataa[:,1],
                                                     bffer,
                                                     minlo, maxlo,
                                                     minla, maxla)
        if qual==1:
            idxs = self.csec.get_grd_nodes_within_buffer_idl(dataa[:,0],
                                                     dataa[:,1],
                                                     bffer,
                                                     minlo, maxlo,
                                                     minla, maxla)
        if idxs is not None and len(idxs):
            boo = numpy.zeros_like(dataa[:, 0], dtype=int)
            boo[idxs[0]] = 1
            self.moho = numpy.squeeze(dataa[idxs,:])


    def set_litho_moho_depth(self, filename, bffer=100.):
        """
        :parameter filename:
            The name of the file containing the LITHO model
        :return:
            A numpy array
        """
        print('setting litho/moho')
        datal = []
        for line in open(filename, 'r'):
            xx = re.split('\s+', re.sub('\s+$', '', re.sub('^\s+', '', line)))
            datal.append([float(val) for val in xx])
        dataa = numpy.array(datal)
        minlo, maxlo, minla, maxla, qual = self.csec.get_mm()
        if qual==0:
            minlo, maxlo, minla, maxla, qual = self.csec.get_mm(2.0)
            idxl = self.csec.get_grd_nodes_within_buffer(dataa[:,0],
                                                     dataa[:,1],
                                                     bffer,
                                                     minlo, maxlo,
                                                     minla, maxla)
        if qual==1:
            idxl = self.csec.get_grd_nodes_within_buffer_idl(dataa[:,0],
                                                     dataa[:,1],
                                                     bffer,
                                                     minlo, maxlo,
                                                     minla, maxla)
        if idxl is not None and len(idxl):
            boo = numpy.zeros_like(dataa[:, 0], dtype=int)
            boo[idxl[0]] = 1
            self.litho = numpy.squeeze(dataa[idxl,:])

    def set_gcmt(self, filename, bffer=75.):
        """
        :parameter cmt_cat:

        """
        print('setting gcmt')
        parser = ParseNDKtoGCMT(filename)
        cmt_cat = parser.read_file()
        loc = cmt_cat.data['longitude']
        lac = cmt_cat.data['latitude']

        minlo, maxlo, minla, maxla, qual = self.csec.get_mm()
        if qual==0:
            idxs = self.csec.get_grd_nodes_within_buffer(loc,
                                                     lac,
                                                     bffer,
                                                     minlo, maxlo,
                                                     minla, maxla)
        if qual==1:
            idxs = self.csec.get_grd_nodes_within_buffer_idl(loc,
                                                     lac,
                                                     bffer,
                                                     minlo, maxlo,
                                                     minla, maxla)
        if idxs is not None:
            cmt_cat.select_catalogue_events(idxs)
            self.gcmt = cmt_cat

    def set_topo(self, filename, bffer=0.25):
        """
        :parameter filename:
            Name of the grid file containing the topography
        """
        print('setting topo')
        datat = []
        for line in open(filename, 'r'):
            tt = re.split('\s+', re.sub('\s+$', '', re.sub('^\s+', '', line)))
            datat.append([float(val) for val in tt])
        datab = numpy.array(datat)
        minlo, maxlo, minla, maxla, qual = self.csec.get_mm()
        if qual==0:
            minlo, maxlo, minla, maxla, qual = self.csec.get_mm(2.0)
            idxb = self.csec.get_grd_nodes_within_buffer(datab[:,0],
                                                     datab[:,1],
                                                     bffer,
                                                     minlo, maxlo,
                                                     minla, maxla)
        if qual==1:
            idxb = self.csec.get_grd_nodes_within_buffer_idl(datab[:,0],
                                                     datab[:,1],
                                                     bffer,
                                                     minlo, maxlo,
                                                     minla, maxla)
        if idxb is not None and len(idxb):
            boo = numpy.zeros_like(datab[:, 0], dtype=int)
            boo[idxb[0]] = 1
            self.topo = numpy.squeeze(datab[idxb,:])

    def set_volcano(self, filename, bffer=75.):
        """
        :parameter filename:
            Name of the file containing the volcano list
        """
        print('setting volcano')
        fin = open(filename, 'r')
        datav = []
        for line in fin:
            vv = re.split('\s+', re.sub('^\s+', '', line))
            datav.append((float(vv[0]), float(vv[1])))

        vulc = numpy.array(datav)
        minlo, maxlo, minla, maxla, qual = self.csec.get_mm()
        if qual==0:
            idxv = self.csec.get_grd_nodes_within_buffer(vulc[:,0],
                                                     vulc[:,1],
                                                     bffer,
                                                     minlo, maxlo,
                                                     minla, maxla)
        if qual==1:
            idxv = self.csec.get_grd_nodes_within_buffer_idl(vulc[:,0],
                                                     vulc[:,1],
                                                     bffer,
                                                     minlo, maxlo,
                                                     minla, maxla)
        if idxv is not None and len(idxv):
            voo = numpy.zeros_like(vulc[:, 0], dtype=int)
            voo[idxv[0]] = 1
            self.volc = numpy.squeeze(vulc[idxv,:])
        fin.close()
        print(self.volc)

class Trench:
    """
    Subduction trench object

    :parameter axis:
        The vertical projection to the topographic surface of the trench axis.
        It's a numpy.array instance with shape (n,2) or (n,3). In the latter
        case the third value in a row represents the depth
    :parameter float depth:
        It's a constant depth value used when the number of columns in the
        `axis` parameter is 2
    """

    def __init__(self, axis, strike=None, azim=None):
        self.axis = axis
        self.strike = strike
        self.azim = azim

    def resample(self, distance):
        """
        This resamples the trench axis given a certain distance and computes
        the strike at each node.

        :parameter distance:
            The sampling distance [in km
        """
        naxis = rsmpl(self.axis[:, 0], self.axis[:, 1], distance)
        if len(self.axis) < 3:
            raise ValueError('Small array')
        #
        # compute azimuths
        az = numpy.zeros_like(self.axis[:, 0])
        az[1:-1] = azimuth(self.axis[:-2, 0], self.axis[:-2, 1],
                           self.axis[2:, 0], self.axis[2:, 1])
        az[0] = az[1]
        az[-1] = az[-2]
        return Trench(naxis,az)

    def iterate_cross_sections(self, distance, length, azim=[]):
        """
        :parameter distance:
            Distance between traces along the trench axis [in km]
        :parameter length:
            The length of each trace [in km]
        """
        trch = self.resample(distance)
        for idx, coo in enumerate(trch.axis.tolist()):
            if idx < len(trch.axis[:, 1]):
                    yield CrossSection(coo[0],
                                       coo[1],
                                       [length],
                                       [(coo[2]+90)%360])
            else:
                yield
        return


def rsmpl_old(ix, iy, samdst):
    """
    """
    #
    # create line instance
    ix = numpy.array([x+360 if x<0 else x for x in ix])
    trench = Line([Point(x, y) for x, y in zip(ix, iy)])
    rtrench = trench.resample(samdst)
    tmp = [[pnt.longitude, pnt.latitude] for pnt in rtrench.points]
    return numpy.array(tmp)

def rsmpl(ix, iy, sampling_dist ):
    direct = 1
    idx = 0
    #
    #create three lists: one with longitude, one with latitude and one with depth
    lo = list(ix)
    la = list(iy)
    de = list(numpy.zeros_like(ix))
    #
    #initialise the variable used to store the cumulated distance
    cdist = 0.
    #
    #get the azimuth of the first segment on the edge in the given direction
    azim = azimuth(lo[idx], la[idx], lo[idx+direct], la[idx+direct])
    #
    #initialise the list with the resampled nodes
    resampled_cs = [[lo[idx], la[idx], azim]]
    #
    # set the starting point
    slo = lo[idx]
    sla = la[idx]
    sde = de[idx]
    #
    #resampling
    while 1:
        #
        #this is a sanity check
        assert idx <= len(lo)-1
        #
        #check loop exit condition
        if direct > 0 and idx > len(lo)-1:
            break
        #
        # compute the distance between the starting point and the next point on the profile
        segment_len = distance(slo, sla, sde, lo[idx+direct], la[idx+direct],                               de[idx+direct])
        #
        #search for the point
        if cdist+segment_len > sampling_dist:
            #
            # this is the lenght of the last segment-fraction needed to
            # obtain the sampling distance
            delta = sampling_dist - cdist
            #
            # add a new point to the cross section
            pnts = npoints_towards(slo, sla, sde, azim, delta, 0., 2)
            #
            #update the starting point
            slo = pnts[0][-1]
            sla = pnts[1][-1]
            sde = pnts[2][-1]
            resampled_cs.append([slo,sla,azim])
            #
            #reset the cumulative distance
            cdist = 0.
        else:
            cdist += segment_len
            idx += direct
            slo = lo[idx]
            sla = la[idx]
            sde = de[idx]
            #
            # get azimuth of the profile
            if idx < len(lo)-1:
                azim = azimuth(lo[idx], la[idx],
                               lo[idx+direct], la[idx+direct])
            else:
                break
    return numpy.array(resampled_cs)


def rsmpl_unsure(ix, iy, sampling_dist):
    direct = 1
    idx = 0
    #
    # create three lists: one with longitude, one with latitude and one with
    # depth
    lo = list(ix)
    la = list(iy)
    de = list(numpy.zeros_like(ix))
    #
    # initialise the variable used to store the cumulated distance
    cdist = 0.
    #
    # set the starting point
    slo = lo[idx]
    sla = la[idx]
    sde = de[idx]
    #
    # get the azimuth of the first segment on the edge in the given direction
    azim = azimuth(lo[idx], la[idx], lo[idx+direct], la[idx+direct])
    #
    # initialise the list with the resampled nodes
    resampled_cs = [[lo[idx], la[idx], azim]]
    #
    # resampling
    while 1:
        #
        # this is a sanity check
        assert idx <= len(lo)-1
        #
        # check loop exit condition
        if direct > 0 and idx > len(lo)-1:
            break
        #
        # compute the distance between the starting point and the next point
        # on the profile
        segment_len = distance(slo, sla, sde, lo[idx+direct], la[idx+direct],
                               de[idx+direct])
        #
        # search for the point
        if cdist+segment_len > sampling_dist:
            #
            # this is the lenght of the last segment-fraction needed to
            # obtain the sampling distance
            delta = sampling_dist - cdist
            #
            # add a new point to the cross section
            pnts = npoints_towards(slo, sla, sde, azim, delta, 0., 2)
            #
            # update the starting point
            slo = pnts[0][-1]
            sla = pnts[1][-1]
            sde = pnts[2][-1]
            resampled_cs.append([slo, sla, azim])
            #
            # reset the cumulative distance
            cdist = 0.
        else:
            cdist += segment_len
            idx += direct
            slo = lo[idx]
            sla = la[idx]
            sde = de[idx]
            #
            # get the azimuth of the profile
            if idx < len(lo)-1:
                azim = azimuth(lo[idx], la[idx],
                               lo[idx+direct], la[idx+direct])
            else:
                break
    #code.interact(local=locals())
    return numpy.array(resampled_cs)


class CrossSection:
    """
    :parameter float olo:
        origin longitude
    :parameter float ola:
        origin latitude
    :parameter length:
        Length of each section [km]. If it is a float it's a single segment
        section if instead it's a list the section will contain as many
        segments as the number of elements in the list.
    :parameter float strike:
        Strike of each section [in decimal degrees]. Data type as per 'length'
        description.
    """

    def __init__(self, olo, ola, length, strike, ids='cs'):
        self.length = length
        self.strike = strike
        self.olo = olo
        self.ola = ola
        self.plo = []
        self.pla = []
        self.ids = ids
        self._set_vertexes()

    def get_mm(self, delta=0.0):
        """
        Get min and maximum values
        """
        lomin = min(self.plo) - delta
        if lomin < -180:
            lomin += 360
	#
        lomax = max(self.plo) + delta
        if lomax > 180:
            lomax -= 360
	#
        lamin = min(self.pla) - delta
        if lamin < -90:
            raise ValueError('Latitude lower than -90')
	#
        lamax = max(self.pla) + delta
        if lamax > 90:
            raise ValueError('Latitude greater than 90')
        #
        qual = 0
        if ((lomin/lomax)<0) & (max([lomin,lomax])>150.):
            qual = 1
        return lomin, lomax, lamin, lamax, qual

    def split_at_idl(self):
        """
        use when a line crosses the international dateline -> divides the line into
        two segments that meet at -180/180 degrees longitude
        """
        #discretize line along ellipsoid and find where it gets closest to idl
        g = Geod(ellps='sphere')
        lonlats = g.npts(self.plo[0],self.pla[0],self.plo[1],self.pla[1],10000)
        modlons = numpy.array(lonlats)
        mdlo1 = abs(modlons[:,0]-180.)
        indlo1 = numpy.argmin(mdlo1)

        #create two lines/subsegments of original line that meet idl
        linplo1 = [-180.,self.plo[0]]; linpla1 = [lonlats[indlo1][1],self.pla[0]]
        linplo2 = [self.plo[1],180]; linpla2 = [self.pla[1],lonlats[indlo1][1]]
        line1 = Line([Point(lo, la) for lo, la in zip(linplo1,linpla1)])
        line2 = Line([Point(lo, la) for lo, la in zip(linplo2,linpla2)])

        return line1, line2, lonlats[indlo1][1]


    def _set_vertexes(self):
        self.plo.append(self.olo)
        self.pla.append(self.ola)
        for lngh, strk in zip(self.length, self.strike):
            tlo, tla = point_at(self.plo[-1], self.pla[-1], strk, lngh)
            self.plo.append(tlo)
            self.pla.append(tla)

    def get_eqks_within_buffer(self, catalogue, buffer_distance):
        """
        :parameter catalogue:
            An instance of :class:`hmtk.catalogue.Catalogue`
        :parameter buffer_distance:
            Horizontal buffer_distance used to select earthquakes included in
            the catalogue [in km]
        """
        xg = catalogue.data['longitude']
        yg = catalogue.data['latitude']
        line = Line([ Point(lo, la) for lo, la in zip(self.plo, self.pla) ])
        coo = [ (lo, la) for lo, la in zip(xg,yg) ]
        dst = get_min_distance(line, numpy.array(coo))
        return numpy.nonzero(abs(dst) <= buffer_distance)

    def get_eqks_within_buffer_idl(self, catalogue, buffer_distance):
        """
        :parameter catalogue:
            An instance of :class:`hmtk.catalogue.Catalogue`
        :parameter buffer_distance:
            Horizontal buffer_distance used to select earthquakes included in
            the catalogue [in km]
        """
        xg = catalogue.data['longitude']
        yg = catalogue.data['latitude']
        line1,line2,center = self.split_at_idl()

        coo = [ (lo, la) for lo, la in zip(xg,yg) ]
        dst1 = get_min_distance(line1, numpy.array(coo))
        dst2 = get_min_distance(line2, numpy.array(coo))
        keep1 = numpy.nonzero(abs(dst1) <= buffer_distance)
        keep2 = numpy.nonzero(abs(dst2) <= buffer_distance)
        keep = numpy.concatenate((keep1,keep2),axis=1)
        return keep

    def get_grd_nodes_within_buffer(self, x, y, buffer_distance,
                                    minlo, maxlo, minla, maxla):

        """
        :parameter x:
            An iterable containing the longitudes of the points defining the
            polyline
        :parameter y:
            An iterable containing the latitudes of the points defining the
            polyline
        :parameter buffer_distance:
            Horizontal buffer_distance used to select earthquakes included in
            the catalogue [in km]
        :parameter minlo:
        :parameter minla:
        :parameter maxlo:
        :parameter maxla:
        """
        line = Line([Point(lo, la) for lo, la in zip(self.plo, self.pla)])
        idxs = numpy.nonzero((x > minlo) & (x < maxlo) &
                             (y > minla) & (y < maxla))
        xs = x[idxs[0]]
        ys = y[idxs[0]]
        coo = [(lo, la) for lo, la in zip(list(xs), list(ys))]
        if len(coo):
            dst = get_min_distance(line, numpy.array(coo))
            return idxs[0][abs(dst) <= buffer_distance]
        else:
            print('   Warning: no nodes found around the cross-section')
            return None

    def get_grd_nodes_within_buffer_idl(self, x, y, buffer_distance,
                                    minlo=-180, maxlo=180, minla=-90, maxla=90):
        """
        :parameter x:
            An iterable containing the longitudes of the points defining the
            polyline
        :parameter y:
            An iterable containing the latitudes of the points defining the
            polyline
        :parameter buffer_distance:
            Horizontal buffer_distance used to select earthquakes included in
            the catalogue [in km]
        :parameter minlo:
        :parameter minla:
        :parameter maxlo:
        :parameter maxla:
        """
        line1,line2,center = self.split_at_idl()
        padding = 2.0
        idxs1 = numpy.nonzero((x > -180.) & (x < (self.plo[0]+padding)) &
                (y < (center+padding)) & (y > (self.pla[0]-padding)))
        xs1 = x[idxs1[0]]
        ys1 = y[idxs1[0]]
        coo1 = [ (lo, la) for lo, la in zip(list(xs1), list(ys1)) ]
        set1 = []
        if len(coo1):
            dst1 = get_min_distance(line1, numpy.array(coo1))
            set1 = idxs1[0][abs(dst1) <= buffer_distance]

        idxs2 = numpy.nonzero((x < 180.) & (x > (self.plo[1]-padding)) &
                (y < (center+padding)) & ((y > self.pla[1]-padding)))
        xs2 = x[idxs2[0]]
        ys2 = y[idxs2[0]]
        coo2 = [ (lo, la) for lo, la in zip(list(xs2), list(ys2)) ]
        set2 = []
        if len(coo2):
            dst2 = get_min_distance(line2, numpy.array(coo2))
            set2 = idxs2[0][abs(dst2) <= buffer_distance]

        if (len(set1)+len(set2))>0:
            use_inds = numpy.concatenate((set1,set2),axis=0)
            return use_inds
        else:
            print ('   Warning: no nodes found around the cross-section')
            return None

def get_min_distance(line, pnts):
    """
    Get distances between a line and a set of points

    :parameter line:
        An instance of :class:`openquake.hazardlib.geo.line.Line`
    :parameter pnts:
        A nx2 array
    """
    #
    #
    assert isinstance(pnts, numpy.ndarray)
    coo = numpy.array([(pnt.longitude, pnt.latitude) for pnt in line.points])
    #
    # this handles the case of a multiine
    if len(coo[:, 0]) > 2:
        cx = numpy.stack((coo[:-1, 0], coo[1:, 0]))
    else:
        cx = [coo[:, 0]]
    if len(coo[:, 0]) > 2:
        cy = list(numpy.stack((coo[:-1, 1], coo[1:, 1])))
    else:
        cy = [
         coo[:, 1]]
    #
    # calculate distances
    distances = numpy.zeros_like(pnts[:, 0])
    distances[:] = 1e+100
    for segx, segy in zip(cx, cy):
        sdx = segx[1] - segx[0]
        sdy = segy[1] - segy[0]
        pdx = segx[0] - pnts[:, 0]
        pdy = segy[0] - pnts[:, 1]
        dot1 = sdx * pdx + sdy * pdy
        pdx = segx[1] - pnts[:, 0]
        pdy = segy[1] - pnts[:, 1]
        dot2 = sdx * pdx + sdy * pdy
        idx = numpy.nonzero((numpy.sign(dot1) < 0) & (numpy.sign(dot2) > 0))
        dst = min_distance_to_segment(segx, segy, pnts[idx[0], 0],
                                      pnts[idx[0], 1])
        distances[idx[0]] = dst
    return distances

