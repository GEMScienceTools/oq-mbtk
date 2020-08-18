import os
import sys
import subprocess
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

class HMTKBaseMap(object):
    '''
    Class to plot the spatial distribution of events based in the Catalogue
    imported from openquake.hmtk.
    '''

    def __init__(self, config, projection='-JM15', #filename=None,
                 ax=None, lat_lon_spacing=2.):
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

        if self.config['title']:
            self.title = config['title']
        else:
            self.title = None

        self.ll_spacing = lat_lon_spacing
        self.fig = None
        self.ax = '-Bx{} -By{}'.format(self.ll_spacing, self.ll_spacing)
        self.m = None
        
        self.J = projection
        self.R = '-R{}/{}/{}/{}'.format(config['min_lon'], 
                                        config['max_lon'],
                                        config['min_lat'],
                                        config['max_lat'])

        self._build_basemap()

        if not os.path.exists('gmt'):
            os.makedirs('gmt')
            

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

    def add_catalogue(self, cat, scale=0.05, cpt_fle="gmt/tmp.cpt"):
        '''
        adds catalogue to map
        '''

        deps = cat.data['depth']
        zmax = max(deps)

        lats = cat.data['latitude']
        lons = cat.data['longitude']
        mags_raw = cat.data['magnitude']
        mags = [scale*10**(-1.5+m*0.3) for m in mags_raw]
        
        df = pd.DataFrame({'lo':lons, 'la':lats, 'd':deps, 'm':mags})
        cat_tmp = 'gmt/cat_tmp.csv'
        df.sort_values(by=['m']).to_csv(cat_tmp, index = False, header = False)

        if cpt_fle == "gmt/tmp.cpt":
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

        fname = 'gmt/legend.csv'
        fou = open(fname, 'w')
        #fou.write("H 11p,Helvetica-Bold Legend\n")
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
        
        filename = 'gmt/mtkAreaSource.csv'
        if os.path.isfile(filename):
            with open(filename,'a') as f:
                f.write('>>')
                for lo,la in zip(lons,lats):
                    f.write('{},{}\n'.format(lo,la))
        else:
            np.savetxt('mtkAreaSource.csv', np.c_[lons,lats])
            self.cmds.append('gmt plot {} -L -Wthick,{}'.format(filename, border))

    def _plot_point_source(self):
        pass

    def _plot_simple_fault(self):
        pass

    def _plot_complex_fault(self):
        pass


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
#        if not overlay:
#            plt.show()

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
        
        filename: string ending in .pdf
        '''
        if filename != None:
            fname = self.cmds[0] + ' gmt/' + filename
            if fname[-4:] != '.pdf':
                fname = fname + '.pdf'
            
            self.cmds[0] = self.cmds[0].replace(self.cmds[0], fname)
        else:
            fname = 'gmt/map.pdf'
            self.cmds[0] = self.cmds[0] + ' ' + fname

        # remove any old instances of gmt end. necessary in case 
        # plotting occurs at differt stages
        self.cmds=[x for x in self.cmds if x != "gmt end"]
        self.cmds.append("gmt end")

        for cmd in self.cmds:
            if verb == 1:
                print(cmd)
            out = subprocess.call(cmd, shell=True)

        print("Map saved to {}.".format(fname))

    def save_gmt_script(self, filename="gmt/gmt_plotter.sh"):
        '''
        saves the gmt plotting commands as a shell script
        '''

        if self.cmds[-1] != "gmt end":
            self.cmds.append("gmt end")
        
        with open(filename,'w') as f:
            f.write('\n'.join(self.cmds))

        print("GMT script written to {}.".format(filename))

    def show(self):
        '''
        Show the pdf in ipython
        '''
        #currently this does not work
        fi = self.title.replace(' ','_')+'.pdf'
        WImage(filename=fi)


