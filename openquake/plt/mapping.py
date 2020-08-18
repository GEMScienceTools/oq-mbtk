import os
import sys
import subprocess
import shutil
import pandas as pd
import numpy as np
from openquake.baselib import sap
from wand.image import Image as WImage
from openquake.hmtk.sources.area_source import mtkAreaSource
from openquake.hmtk.sources.point_source import mtkPointSource
#from openquake.hmtk.plotting.beachball import Beach
#from openquake.hmtk.plotting.plotting_utils import DISSIMILAR_COLOURLIST
from openquake.hmtk.sources.simple_fault_source import mtkSimpleFaultSource
from openquake.hmtk.sources.complex_fault_source import mtkComplexFaultSource

def _fault_polygon_from_mesh(source):
    # Mesh
    upper_edge = np.column_stack([source.geometry.mesh.lons[1],
                                  source.geometry.mesh.lats[1],
                                  source.geometry.mesh.depths[1]])
    lower_edge = np.column_stack([source.geometry.mesh.lons[-1],
                                  source.geometry.mesh.lats[-1],
                                  source.geometry.mesh.depths[-1]])
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

        self._build_basemap()


        # set starter values that may be replaced when making the colors
        self.max_cf_depth = 1000

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

        cmds = []

        cmds.append("gmt begin")# {}".format(self.filename))
        tmp = "gmt basemap {} {} -BWSne".format(self.R, self.J)
        tmp += " {}".format(self.ax)
        cmds.append(tmp)

        cmds.append("gmt coast -Df {} {} -Wthin -Gwheat".format(self.R, self.J))
        
        self.cmds = cmds 

        return self.cmds

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

        tmp = "gmt plot {} -Sc -C{} -Wthinnest,black".format(cat_tmp,cpt_fle)
        self.cmds.append(tmp)
        self.cmds.append('gmt colorbar -DJBC -Ba{}+l"Depth (km)" -C{}'.format('100', cpt_fle))
        
        self._add_legend(mags_raw, scale)

    def _add_legend(self, mags, scale):
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
        lons = np.hstack([source.geometry.lons, source.geometry.lons[0]])
        lats = np.hstack([source.geometry.lats, source.geometry.lats[0]])
        
        filename = '{}/mtkAreaSource.csv'.format(self.out)
        add_plot_line = self.mk_plt_csv(lons, lats, filename, lines=1)

        if add_plot_line == 1:
            self.cmds.append('gmt plot {} -L -Wthick,{}'.format(filename, border))

    def _plot_point_source(self, source, pointsize=0.08):
        x, y = self.m(source.geometry.longitude, source.geometry.latitude)

        filename = '{}/mtkPointSource.csv'.format(self.out)

        add_plot_line = self.mk_plt_csv(lons, lats, filename)

        if add_plot_line == 1:
            self.cmds.append('gmt plot {} -Ss{}'.format(filename, pointsize))


    def _plot_simple_fault(self, source):
        trace_lons = np.array([pnt.longitude
                               for pnt in source.fault_trace.points])
        trace_lats = np.array([pnt.latitude
                               for pnt in source.fault_trace.points])
        surface_projection = _fault_polygon_from_mesh(source)

        # First make surface projection file and command
        x, y = self.m(surface_projection[:, 0], surface_projection[:, 1])

        filename = '{}/mtkSimpleFaultProjection.csv'.format(self.out)
        add_plot_line = self.mk_plt_csv(lons, lats, filename, lines=1)
        
        if add_plot_line == 1:
            self.cmds.append('gmt plot {} -t50 -Ggray'.format(filename))

        # then fault trace 
        x, y = self.m(trace_lons, trace_lats)
        filename = '{}/mtkSimpleFaultTrace.csv'.format(self.out)
        add_plot_line = self.mk_plt_csv(lons, lats, filename, lines=1)
        
        if add_plot_line == 1:
            self.cmds.append('gmt plot {} -Wthick,red'.format(filename))

    def _plot_complex_fault(self, source):
        max_depth = 600.

        top_edge = np.column_stack([source.geometry.mesh.lons[0],
                                    source.geometry.mesh.lats[0]])

        bottom_edge = np.column_stack([source.geometry.mesh.lons[-1][::-1],
                                       source.geometry.mesh.lats[-1][::-1]])
        outline = np.vstack([top_edge, bottom_edge, top_edge[0, :]])
        lons = source.geometry.mesh.lons.flatten()
        lats = source.geometry.mesh.lats.flatten()
        depths = source.geometry.mesh.depths.flatten()
#        norm = Normalize(vmin=min_depth, vmax=max_depth)

        filename = '{}/mtkComplexFaultPoints.csv'.format(self.out)
        add_plot_line = self.mk_plt_csv(lons, lats, filename, color_column=depths)

        if add_plot_line == 1:
            # Making cpt
            cpt_fle = "{}/cf_tmp.cpt".format(self.out)
            self.cmds.append("gmt makecpt -Cjet -T0/{}/1 -Q -D > {:s}".format(10, cpt_fle))

            #self.cmds.append('gmt plot {} -Ss0.5 -Ggreen '.format(filename))
            
            self.cmds.append('gmt plot {} -C{} -Ss0.5 '.format(filename, cpt_fle))

        # Plot border
        #x2, y2 = self.m(outline[:, 0], outline[:, 1])
        #self.m.plot(x2, y2, border, linewidth=border_width)

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

        for source in model.sources:
            if isinstance(source, mtkAreaSource):
                self._plot_area_source(source)
            elif isinstance(source, mtkPointSource):
                self._plot_point_source(source)#, point_marker, point_size)
            elif isinstance(source, mtkComplexFaultSource):
                self._plot_complex_fault(source)#, area_border, border_width,
                                         #min_depth, max_depth, alpha)
            elif isinstance(source, mtkSimpleFaultSource):
                self._plot_simple_fault(source)#, area_border, border_width)
            else:
                pass

    def add_colour_scaled_points(self):
        pass

    def add_self_scaled_points(self):
        pass

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

    def add_focal_mechanism(self):
        pass

    def add_catalogue_cluster(self):
        pass

    def savemap(self, filename=None, verb=0):
        '''
        Saves map
        '''

        # file must be a pdf. set path and modify accordingly 

        if filename != None:
            begin = 'gmt begin {}/{}'.format(self.out, filename)
            if begin[-4:] != '.pdf':
                begin = begin + '.pdf'
            
            self.cmds[0] = begin 

        else:
            begin = 'gmt begin {}/map.pdf'.format(self.out)
            self.cmds[0] = begin

        # remove any old instances of gmt end, then re-add
        # necessary in case plotting occurs at differt stages

        self.cmds=[x for x in self.cmds if x != "gmt end"]
        self.cmds.append("gmt end")

        for cmd in self.cmds:
            if verb == 1:
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

