import os
import sys
import subprocess
import shutil
import pandas as pd
import numpy as np
from openquake.baselib import sap
from wand.image import Image as WImage
from openquake.hazardlib.source.complex_fault import ComplexFaultSurface
from openquake.hazardlib.source.simple_fault import SimpleFaultSurface

def _fault_polygon_from_mesh(surface):
    # Mesh
    upper_edge = np.column_stack([surface.mesh.lons[0],
                                  surface.mesh.lats[0],
                                  surface.mesh.depths[0]])
    lower_edge = np.column_stack([surface.mesh.lons[-1],
                                  surface.mesh.lats[-1],
                                  surface.mesh.depths[-1]])
    return np.vstack([upper_edge, np.flipud(lower_edge), upper_edge[0, :]])


class HMTKBaseMap(object):
    '''
    Class to plot the spatial distribution of events based in the Catalogue
    imported from openquake.hmtk.
    '''

    def __init__(self, config, projection='-JM15', output_folder='gmt',
                 ax=None, lat_lon_spacing=2., overwrite=False):
        """
        :param dict config:
            Configuration parameters of the algorithm, containing the
            following information -
                'min_lat' Minimum value of latitude (in degrees, float)
                'max_lat' Minimum value of longitude (in degrees, float)
                (min_lat, min_lon) Defines the inferior corner of the map
                'min_lon' Maximum value of latitude (in degrees, float)
                'max_lon' Maximum value of longitude (in degrees, float)
                (min_lon, max_lon) Defines the upper corner of the map
        :param str title:
            Title string
        """
        
        self.config = config
        self.out = output_folder

        if self.config['title']:
            self.title = config['title']
        else:
            self.title = None

        self.ll_spacing = lat_lon_spacing
        self.fig = None
        self.ax = '-Bx{} -By{}'.format(self.ll_spacing, self.ll_spacing)
        
        self.J = projection
        self.R = '-R{}/{}/{}/{}'.format(config['min_lon'], 
                                        config['max_lon'],
                                        config['min_lat'],
                                        config['max_lat'])

#        self.cmds = []
        self._build_basemap()


        # set starter values that may be replaced when making the colors
        self.max_cf_depth = 1000
        self.max_sf_depth = 1000

        # create the output directory. Check if it exists, whether overwrite 
        # is allowed, rm dir contents or fail

        if os.path.exists(self.out):
            if overwrite == True:
                shutil.rmtree(self.out)
            else:
                warning = "{} directory already exists!\n".format(self.out)
                warning += "Set overwrite=True or change the output path."
                raise ValueError(warning)

        os.makedirs(self.out)
            

    def _build_basemap(self):
        '''
        Creates the map according to the input configuration
        '''


        self.cmds = []
        self.cmds.append("gmt begin")
        tmp = "gmt basemap {} {} -BWSne".format(self.R, self.J)
        tmp += " {}".format(self.ax)
        self.cmds.append(tmp)

        self.cmds.append("gmt coast -Df {} {} -Wthin -Gwheat".format(self.R, self.J))
        

    def add_catalogue(self, cat, scale=0.05, cpt_file="tmp.cpt"):
        '''
        adds catalogue to map
        '''
        cpt_fle = "{}/{}".format(self.out, cpt_file)

        deps = cat.data['depth']
        zmax = max(deps)

        lats = cat.data['latitude']
        lons = cat.data['longitude']
        mags_raw = cat.data['magnitude']
        mags = [scale*10**(-1.5+m*0.3) for m in mags_raw]
        
        df = pd.DataFrame({'lo':lons, 'la':lats, 'd':deps, 'm':mags})
        cat_tmp = '{}/cat_tmp.csv'.format(self.out)
        df.sort_values(by=['m']).to_csv(cat_tmp, index = False, header = False)

        if cpt_fle == "{}/tmp.cpt".format(self.out):
            self.cmds.append("gmt makecpt -Cjet -T0/2.7/30+n -Q -D > \
                             {}".format(cpt_fle))

        space = np.floor(abs(min(data)-max(data))/3)
        tmp = "gmt plot {} -Sc -C{} -Wthinnest,black".format(cat_tmp,cpt_fle)
        self.cmds.append(tmp)
        self.cmds.append('gmt colorbar -DJBC -Ba{}+l"Depth (km)" -C{}'.format(space, cpt_fle))
        
        self._add_legend_catalogue(mags_raw, scale)

    def _add_legend_catalogue(self, mags, scale):
        '''
        adds legend for catalogue seismicity
        '''

        fname = '{}/legend.csv'.format(self.out)
        fou = open(fname, 'w')
        fou.write("L 9p R Magnitude\n")
        fmt = "S 0.4i c {:.4f} - 0.0c,black 2.0c {:.0f} \n"


        minmag = np.floor(min(mags))
        maxmag = np.ceil(max(mags))

        ms = np.arange(minmag,maxmag+1)

        for m in ms:
            sze = scale*10**(-1.5+m*0.3)
            fou.write(fmt.format(sze, m))

        fou.close()

        tmp = "gmt legend {} -DJMR -C0.3c ".format(fname)
        tmp += "--FONT_ANNOT_PRIMARY=9p"
        self.cmds.append(tmp)
        
            
    def _plot_area_source(self, source, border='blue'):
        poly = source.polygon
        lons = np.append(poly.lons, poly.lons[0])
        lats = np.append(poly.lats, poly.lats[0])
        
        filename = '{}/mtkAreaSource.csv'.format(self.out)
        add_plot_line = self.mk_plt_csv(lons, lats, filename, lines=1)

        if add_plot_line == 1:
            self.cmds.append('gmt plot {} -L -Wthick,{}'.format(filename, border))

    def _plot_point_source(self, source, pointsize=0.5):

        lons = source.location.longitude
        lats = source.location.latitude

        filename = '{}/mtkPointSource.csv'.format(self.out)

        add_plot_line = self.mk_plt_csv(np.array([lons]), np.array([lats]), filename)

        if add_plot_line == 1:
            self.cmds.append('gmt plot {} -Ss{} -Gred'.format(filename, pointsize))


    def _plot_simple_fault(self, source):
        trace_lons = np.array([pnt.longitude
                               for pnt in source.fault_trace.points])
        trace_lats = np.array([pnt.latitude
                               for pnt in source.fault_trace.points])

        fault_surface = SimpleFaultSurface.from_fault_data(
                source.fault_trace, source.upper_seismogenic_depth,
                source.lower_seismogenic_depth, source.dip, source.rupture_mesh_spacing)

        outline = _fault_polygon_from_mesh(fault_surface)

        lons = fault_surface.mesh.lons.flatten()
        lats = fault_surface.mesh.lats.flatten()
        depths = fault_surface.mesh.depths.flatten()

        self.max_sf_depth = max(depths) if max(depths) < self.max_sf_depth \
                else self.max_sf_depth

        filename = '{}/mtkSimpleFaultSurface.csv'.format(self.out)
        add_plot_line = self.mk_plt_csv(lons, lats, filename, 
                                        color_column=depths, lines=1)
        
        if add_plot_line == 1:
            cpt_fle = "{}/sf_tmp.cpt".format(self.out)
            self.cmds.append("gmt makecpt -Cjet -T0/{}/30+n > {:s}".format(
                self.max_sf_depth*1.2, cpt_fle))

            self.cmds.append('gmt plot {} -C{} -Ss0.1 '.format(filename, cpt_fle))
            self.cmds.append('gmt colorbar -DJBC -Ba{}+l"Depth (km)" -C{}'.format(
                '10', cpt_fle))

        filename = '{}/mtkSimpleFaultProjection.csv'.format(self.out)
        add_plot_line = self.mk_plt_csv(outline[:, 0], outline[:, 1], filename, lines=1)
        if add_plot_line == 1:
            self.cmds.append('gmt plot {} -Wblack'.format(filename))
        # then fault trace 
        filename = '{}/mtkSimpleFaultTrace.csv'.format(self.out)
        add_plot_line = self.mk_plt_csv(trace_lons, trace_lats, filename, lines=1)
        
        if add_plot_line == 1:
            self.cmds.append('gmt plot {} -Wthick,red'.format(filename))

    def _plot_complex_fault(self, source):

        fault_surface = ComplexFaultSurface.from_fault_data(
            source.edges, source.rupture_mesh_spacing)

        outline = _fault_polygon_from_mesh(fault_surface)

        lons = fault_surface.mesh.lons.flatten()
        lats = fault_surface.mesh.lats.flatten()
        depths = fault_surface.mesh.depths.flatten()

        self.max_cf_depth = max(depths) if max(depths) < self.max_cf_depth \
                else self.max_cf_depth

        filename = '{}/mtkComplexFaultPoints.csv'.format(self.out)
        add_plot_line = self.mk_plt_csv(lons, lats, filename, color_column=depths)

        if add_plot_line == 1:
            # Making cpt
            cpt_fle = "{}/cf_tmp.cpt".format(self.out)
            self.cmds.append("gmt makecpt -Cjet -T0/{}/2> {:s}".format(
                self.max_cf_depth, cpt_fle))

            self.cmds.append('gmt plot {} -C{} -Ss0.1 '.format(filename, cpt_fle))
            self.cmds.append('gmt colorbar -DJBC -Ba{}+l"Depth (km)" -C{}'.format(
                '10', cpt_fle))

        filename = '{}/mtkComplexFaultOutline.csv'.format(self.out)
        add_plot_line = self.mk_plt_csv(outline[:, 0], outline[:, 1], filename, lines=1)
       
        if add_plot_line == 1:
            self.cmds.append('gmt plot {} -Wthick,black'.format(filename))

    def mk_plt_csv(self, lons, lats, filename, color_column=None, lines=None):

        if lines == 1:
            lons = np.append(lons,'>>')
            lats = np.append(lats,'nan')
            if color_column is not None:
                color_column = np.append(color_column, 'nan')

        if color_column is None:
            d = {'lons': lons, 'lats': lats}
            df = pd.DataFrame(data=d)
        else:
            d = {'lons': lons, 'lats': lats, 'zs': color_column}
            df = pd.DataFrame(data=d)

        add_plot_line = 0 if os.path.isfile(filename) else 1

        with open(filename,'a') as f:
             df.to_csv(f, header=False, index=False)
             
        return add_plot_line

    def add_source_model(self, model):

        for grp in model.src_groups:
            for source in grp:
                if type(source).__name__ == 'AreaSource':
                    self._plot_area_source(source)
                elif type(source).__name__ == 'PointSource': 
                    self._plot_point_source(source)#, point_marker, point_size)
                elif type(source).__name__ == 'ComplexFaultSource':
                    self._plot_complex_fault(source)#, area_border, border_width,
                                             #min_depth, max_depth, alpha)
                elif type(source).__name__ == 'SimpleFaultSource':
                    self._plot_simple_fault(source)#, area_border, border_width)
                else:
                    pass

    def add_colour_scaled_points(self, longitude, latitude, data, label="Data value",
            shape="-Ss", size=0.3, logscale=False):
        '''
        Adds xy data colored by some specified data value

        :param str shape: 
            shape of the plotted data. See GMT documentation. Must start
            with -S
        :param str label:
            Data label for the colorbar

        '''
#        if not norm:
#            norm = Normalize(vmin=np.min(data), vmax=np.max(data))


        cpt_fle = "{}/tmp_col_dat.cpt".format(self.out)
        if logscale:
            self.cmds.append("gmt makecpt -Cjet -T{}/{}/30+n -Q -D > \
                             {}".format(min(data), max(data), cpt_fle))
        else:
            self.cmds.append("gmt makecpt -Cjet -T{}/{}/30+n -D > \
                             {}".format(min(data), max(data), cpt_fle))


        df = pd.DataFrame({'lo':longitude, 'la':latitude, 'c':data})
        dat_tmp = '{}/tmp_dat_col.csv'.format(self.out)
        df.to_csv(dat_tmp, index = False, header = False)

        space = np.floor(abs(min(data)-max(data))/3)
        self.cmds.append('gmt plot {} {}{} -C{}'.format(dat_tmp, shape, size, cpt_fle))
        self.cmds.append('gmt colorbar -DJBC -Ba{}+l{} -C{}'.format(space, label, cpt_fle))

    def add_size_scaled_points(self, longitude, latitude, data, shape='-Ss',
            logplot=False, color='blue', smin=0.01, coeff=1.0, sscale=2.0, label=''):

        if logplot:
            data = np.log10(data.copy())

        size = smin + coeff * data ** sscale

        df = pd.DataFrame({'lo':longitude, 'la':latitude, 's':size})
        dat_tmp = '{}/tmp_dat_size.csv'.format(self.out)
        df.to_csv(dat_tmp, index = False, header = False)


        self.cmds.append('gmt plot {} {} -G{} -Wblack'.format(dat_tmp, shape, color))

        mindat = np.floor(min(data))
        maxdat = np.ceil(max(data))
        drange = abs(mindat - maxdat)
        
        ds = np.arange(mindat,maxdat+1,np.ceil(drange/5))
        sz = smin + coeff * ds ** sscale

        self._add_legend_size_scaled(ds, color, sz, shape, label)


    def _add_legend_size_scaled(self, data, color, size, shape, label):
        '''
        adds legend for catalogue seismicity
        '''

        fname = '{}/legend_ss.csv'.format(self.out)
        fou = open(fname, 'w')
        fou.write("L 12p R {}\n".format(label))
        fou.write('G 0.1i\n')
        fmt = "S 0.4i {} {:.4f} {} 0.0c,black 2.0c {:.0f} \n"

        sh = shape.replace('-S','').replace("'",'')

        for dd,ss in zip(data, size):
            fou.write(fmt.format(sh, ss, color, dd))
            fou.write('G 0.2i\n')

        fou.close()

        tmp = "gmt legend {} -DJMR -C0.3c ".format(fname)
        tmp += "--FONT_ANNOT_PRIMARY=9p"
        self.cmds.append(tmp)

    def _select_color_mag(self, mag):
        if (mag > 8.0):
            color = 'k'
        elif (mag < 8.0) and (mag >= 7.0):
            color = 'b'
        elif (mag < 7.0) and (mag >= 6.0):
            color = 'y'
        elif (mag < 6.0) and (mag >= 5.0):
            color = 'g'
        elif (mag < 5.0):
            color = 'm'
        return color

    def add_focal_mechanism(self, filename, mech_format, config=None):
        '''
        string filename:
            the filename containing the gcmt entries
        string mech_format: 
            the format of the file to be plotted. either focal mechanism 
            (mech_format='FM') or seimsic moment tensor (mech_format='MT')
            both using the Harvard CMT convention
        '''

        if mech_format == 'FM':
            mf = 'c'
        elif mech_format == 'MT':
            mf = 'm'
        else:
            fail_error = "mech_format must be either 'FM' or 'MT'; see doc"
            raise ValueError(fail_error)

        if config is not None:
            df = pd.read_csv(filename)
            # TODO: make some other settings... scale by mag, color, don't
            # use label, etc
        else:
            self.cmds.append("gmt psmeca {} -S{}0.5 -t20".format(filename,\
                    mf)


    def add_catalogue_cluster(self):
        pass

    def savemap(self, filename=None, verb=False):
        '''
        Saves map
        '''

        # file must be a pdf. set path and modify accordingly 

        if filename != None:
            begin = 'gmt begin {}/{}'.format(self.out, filename)
            if begin[-4:] != '.pdf':
                begin = begin + '.pdf'
            

        else:
            begin = 'gmt begin {}/map.pdf'.format(self.out)

        self.cmds[0] = begin

        # remove any old instances of gmt end, then re-add
        # necessary in case plotting occurs at differt stages

        self.cmds=[x for x in self.cmds if x != "gmt end"]
        self.cmds.append("gmt end")

        for cmd in self.cmds:
            if verb:
                print(cmd)
            out = subprocess.call(cmd, shell=True)

        print("Map saved to {}.".format(begin.split(' ')[-1]))

    def save_gmt_script(self, filename="gmt_plotter.sh"):
        '''
        saves the gmt plotting commands as a shell script
        '''

        if self.cmds[-1] != "gmt end":
            self.cmds.append("gmt end")

        fname = '{}/{}'.format(self.out, filename)
        
        with open(fname,'w') as f:
            f.write('\n'.join(self.cmds))

        print("GMT script written to {}.".format(fname))

    def show(self):
        '''
        Show the pdf in ipython
        '''
        #TO DO
        pass

