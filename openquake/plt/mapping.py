import os
import sys
import subprocess
import pandas as pd
from openquake.baselib import sap
from wand.image import Image as WImage

class HMTKBaseMap(object):
    '''
    Class to plot the spatial distribution of events based in the Catalogue
    imported from openquake.hmtk.
    '''

    def __init__(self, config, projection='-JM8', filename=None,
                 dpi=300, ax=None, lat_lon_spacing=2.):
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
        self.filename = filename
        if self.filename == None:
            self.filename = 'Map.pdf'

        self.title = config['title']
        self.dpi = dpi
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

    def _build_basemap(self):
        '''
        Creates the map according to the input configuration
        '''

        cmds = []

        cmds.append("gmt begin {}".format(self.filename))
        tmp = "gmt basemap {} {} -BWSne".format(self.R, self.J)
        tmp += " {}".format(self.ax)
        cmds.append(tmp)

        cmds.append("gmt coast -Df {} {} -Wthin -Gwheat".format(self.R, self.J))
        
        self.cmds = cmds 

        return self.cmds

    def add_catalogue(self, cat):
        '''
        adds catalogue to map
        '''

        deps = cat.data['depth']
        zmax = max(deps)

        lats = cat.data['latitude']
        lons = cat.data['longitude']
        mags = [0.05*10**(-1.5+m*0.3) for m in cat.data['magnitude']]
        
        df = pd.DataFrame({'lo':lons, 'la':lats, 'd':deps, 'm':mags})
        df.sort_values(by=['m']).to_csv('cat_tmp.csv',index = False, header=False)

        cpt_fle = "tmp.cpt"
        self.cmds.append("gmt makecpt -Cjet -T0/2.7/30+n -Q -D > \
                          {}".format(cpt_fle))
                          #{}".format(zmax, zmax/5, cpt_fle))

        tmp = "gmt plot {} -Sc -C{} -Wthinnest,black".format('cat_tmp.csv',cpt_fle)
        self.cmds.append(tmp)
        self.cmds.append('gmt colorbar -DJCR -Ba+l"Depth (km)" -C{}'.format(cpt_fle))

    def savemap(self, verb=0):
        '''
        Saves map
        '''

        self.cmds.append("gmt end")

        for cmd in self.cmds:
            if verb == 1:
                print(cmd)
            out = subprocess.call(cmd, shell=True)

    def show(self):
        '''
        Show the pdf in ipython
        '''
        #currently this does not work
        fi = self.title.replace(' ','_')+'.pdf'
        WImage(filename=fi)


