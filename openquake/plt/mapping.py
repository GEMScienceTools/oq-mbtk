import os
import sys
import subprocess
import pandas as pd
import numpy as np
from openquake.baselib import sap
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
    imported from openquake.hmtk using Generic Mapping Tools.

    Initiates a plot and a GMT mapping script 
    '''

    def __init__(self, config, projection='-JM15', output_folder='gmt',
                 lat_lon_spacing=2., overwrite=False):
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
                'title' (optional) map title
        :param str projection:
            String beginning with '-J' that indicates the projection 
            following the GMT syntax 
            http://gmt.soest.hawaii.edu/doc/latest/gmt.html#j-full
        :param str output_folder:
            Directory (relative to working directory) where all outputs 
            will be saved. 
        :param float lat_lon_spacing:
            x- and y- spacing of tick marks along map border
        :param boolean overwrite:
            True means that all files will be overwritten. False requires
            an output_folder name that is not already in use by a directory
        """
        
        self.config = config
        self.out = output_folder
        self.overwrite = overwrite

        # make the output directory if it doesn't exist
        if os.path.exists(self.out):
            pass
        else:
            os.makedirs(self.out)

        # set the title if it is specified
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
        self.gmt_files_list = []


        # initiate integers that may be replaced when making the colors
        self.max_cf_depth = 1000
        self.max_sf_depth = 1000


    def _check_output(self,filename):
        # create the output directory. Check if it exists, whether overwrite 
        # is allowed, rm dir contents or fail

        outfile = os.path.join(self.out, filename)
        if os.path.exists(outfile):
            if self.overwrite == True:
                os.remove(outfile)
            else:
                warning = "output file {}/{} already exists!\n".format(self.out, filename)
                warning += "Set overwrite=True or change the output directory or filename."
                raise ValueError(warning)

            

    def _build_basemap(self):
        '''
        Creates the map according to the input configuration
        '''


        self.cmds = []
        self.cmds.append("gmt begin")
        tmp = 'gmt basemap {} {} -BWSne+t"{}"'.format(self.R, self.J, self.title)
        tmp += " {}".format(self.ax)
        self.cmds.append(tmp)

        self.cmds.append("gmt coast -Di {} {} -Wthin -Gwheat".format(self.R, self.J))
        

    def add_catalogue(self, cat, scale=0.05, cpt_file="tmp.cpt", color_field='depth',
                      logscale=True):
        '''
        adds catalogue to map
        :param cat:
            Earthquake catalogue as instance of 
            :class:`openquake.hmtk.seismicity.catalogue.Catalogue`
        :param float scale:
            Scaling coefficient for symbol size per magnitude.
        :param str cpt_file:
            Name of file (no path) where color pallet with be saved
        :param str color_field:
            Field used to color the symbols. Must correspond to header.
        '''
        cpt_fle = "{}/{}".format(self.out, cpt_file)

        zfield = cat.data[color_field]
        zmax = max(zfield)
        zmin = min(zfield)

        if color_field == 'magnitude' and logscale == True:
            print('Logscale cannot be used with magnitude; setting logscale=False')
            logscale=False

        if color_field == 'depth' and zmin == 0:
            zmin = 1

        lats = cat.data['latitude']
        lons = cat.data['longitude']
        mags_raw = cat.data['magnitude']
        mags = [scale*10**(-1.5+m*0.3) for m in mags_raw]
        
        df = pd.DataFrame({'lo':lons, 'la':lats, 'd':zfield, 'm':mags})
        cat_tmp = '{}/cat_tmp.csv'.format(self.out)
        self.gmt_files_list.append(cat_tmp)

        df.sort_values(by=['m']).to_csv(cat_tmp, index = False, header = False)

        if cpt_fle == "{}/tmp.cpt".format(self.out):
            if logscale is True:
                self.cmds.append("gmt makecpt -Cjet -T{}/{}/30+n -Q -D > \
                                 {}".format(np.log10(zmin), np.log10(zmax), cpt_fle))
            else:
                self.cmds.append("gmt makecpt -Cjet -T{}/{}/30+n -D > \
                                 {}".format(zmin, zmax, cpt_fle))
            self.gmt_files_list.append(cpt_fle)

        space = np.floor(abs(min(zfield)-max(zfield))/4)
        tmp = "gmt plot {} -Sc -C{} -Wthinnest,black".format(cat_tmp,cpt_fle)
        self.cmds.append(tmp)
        self.cmds.append('gmt colorbar -DJBC -Ba{}+l"{}" -C{}'.format(space, 
                                                                      color_field,
                                                                      cpt_fle))
        
        self._add_legend_catalogue(mags_raw, scale)
        self.gmt_files_list.append('{}/legend.csv'.format(self.out))

    def _add_legend_catalogue(self, mags, scale):
        '''
        Called by self.add_catalogue. Adds legend for catalogue seismicity
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
        '''
        Adds area source perimeters to mapping script. 
        :param source:
            area source as instance of 
            :class:`openquake.hazardlib.source.area.AreaSource`
        :param str border:
           color of the area source perimeters
        '''
        poly = source.polygon
        lons = np.append(poly.lons, poly.lons[0])
        lats = np.append(poly.lats, poly.lats[0])
        
        filename = '{}/mtkAreaSource.csv'.format(self.out)
        add_plot_line = self.mk_plt_csv(lons, lats, filename, lines=1)

        if add_plot_line == 1:
            self.gmt_files_list.append(filename)
            self.cmds.append('gmt plot {} -L -Wthick,{}'.format(filename, border))

    def _plot_point_source(self, source, pointsize=0.5):
        '''
        Adds point sources to mapping script. 
        :param source:
            point source as instance of 
            :class:`openquake.hazardlib.source.point.PointSource`
        :param float pointsize:
            sets the size of plotting symbols 
        '''

        lons = source.location.longitude
        lats = source.location.latitude

        filename = '{}/mtkPointSource.csv'.format(self.out)

        add_plot_line = self.mk_plt_csv(np.array([lons]), np.array([lats]), filename)

        if add_plot_line == 1:
            self.cmds.append('gmt plot {} -Ss{} -Gred'.format(filename, pointsize))
            self.gmt_files_list.append(filename)


    def _plot_simple_fault(self, source):
        '''
        Adds simple fault sources to mapping script. 
        :param source:
            simple fault source as instance of 
            :class:`openquake.hazardlib.source.simple_fault.SimpleFaultSource`
        '''

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
            self.gmt_files_list.append(filename)
            cpt_fle = "{}/sf_tmp.cpt".format(self.out)
            self.gmt_files_list.append(cpt_fle)
            self.cmds.append("gmt makecpt -Cjet -T0/{}/30+n > {:s}".format(
                self.max_sf_depth*1.2, cpt_fle))

            self.cmds.append('gmt plot {} -C{} -Ss0.1 '.format(filename, cpt_fle))
            self.cmds.append('gmt colorbar -DJBC -Ba{}+l"Depth (km)" -C{}'.format(
                '10', cpt_fle))

        filename = '{}/mtkSimpleFaultProjection.csv'.format(self.out)
        add_plot_line = self.mk_plt_csv(outline[:, 0], outline[:, 1], filename, lines=1)
        if add_plot_line == 1:
            self.gmt_files_list.append(filename)
            self.cmds.append('gmt plot {} -Wblack'.format(filename))
        # then fault trace 
        filename = '{}/mtkSimpleFaultTrace.csv'.format(self.out)
        add_plot_line = self.mk_plt_csv(trace_lons, trace_lats, filename, lines=1)
        
        if add_plot_line == 1:
            self.gmt_files_list.append(filename)
            self.cmds.append('gmt plot {} -Wthick,red'.format(filename))

    def _plot_complex_fault(self, source):
        '''
        Adds complex fault sources to mapping script. 
        :param source:
            complex fault source as instance of 
            :class:`openquake.hazardlib.source.complex_fault.ComplexFaultSource`
        '''

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
            self.gmt_files_list.append(filename)
            cpt_fle = "{}/cf_tmp.cpt".format(self.out)
            self.gmt_files_list.append(cpt_fle)
            self.cmds.append("gmt makecpt -Cjet -T0/{}/2> {:s}".format(
                self.max_cf_depth, cpt_fle))

            self.cmds.append('gmt plot {} -C{} -Ss0.1 '.format(filename, cpt_fle))
            self.cmds.append('gmt colorbar -DJBC -Ba{}+l"Depth (km)" -C{}'.format(
                '10', cpt_fle))

        filename = '{}/mtkComplexFaultOutline.csv'.format(self.out)
        add_plot_line = self.mk_plt_csv(outline[:, 0], outline[:, 1], filename, lines=1)
       
        if add_plot_line == 1:
            self.gmt_files_list.append(filename)
            self.cmds.append('gmt plot {} -Wthick,black'.format(filename))

    def mk_plt_csv(self, lons, lats, filename, color_column=None, lines=0):
        '''
        creates csv file formatted for GMT to plot catalogue/other xyz data
        :param array lons:
            x coordinates/longitudes of data to be plotted
        :param array lats:
            y coordinates/latitudes of data to be plotted
        :param str filename:
            name of the csv file to save the data to
        :param array color_column:
            values to be used for plot color scaling
        :param lines:
            indicates lines/polygons (1) or points (0)
        '''

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

        #add_plot_line = 0 if os.path.isfile(filename) else 1
        chk = sum([1 if c.find(filename)>0 else 0 for c in self.cmds])
        add_plot_line = 0 if chk > 0 else 1

        with open(filename,'a') as f:
             df.to_csv(f, header=False, index=False)
             
        return add_plot_line

    def add_source_model(self, model):
        '''
        adds source model to mapping script
        :param model:
            a source model as instance of 
            :class:`openquake.hazardlib.nrml.SourceModel`
        '''

        for grp in model.src_groups:
            for source in grp:
                if type(source).__name__ == 'AreaSource':
                    self._plot_area_source(source)
                elif type(source).__name__ == 'PointSource': 
                    self._plot_point_source(source)
                elif type(source).__name__ == 'ComplexFaultSource':
                    self._plot_complex_fault(source)
                elif type(source).__name__ == 'SimpleFaultSource':
                    self._plot_simple_fault(source)
                else:
                    pass

    def add_colour_scaled_points(self, longitude, latitude, data, label='',
            shape="-Ss", size=0.3, logscale=False):
        '''
        Adds xy data (epicenters) colored by some specified data value
        :param array longitude:
            x coordinates/longitudes of data to be plotted
        :param array latitude:
            y coordinates/latitudes of data to be plotted
        :param array data:
            array to be used to color-scale the xy data
        :param str label:
            Data label for the colorbar and plot title. Also used to name tmp file
        :param str shape: 
            shape of the plotted data. Must start with '-S'. Default is a square.
            See GMT documentation.
            https://docs.generic-mapping-tools.org/latest/psxy.html#s
        :param float size:
            size of the plotted symbols
        :param logscale:
            if True, scale colors in log space
        '''
#        if not norm:
#            norm = Normalize(vmin=np.min(data), vmax=np.max(data))


        cpt_fle = "{}/tmp_col_dat.cpt".format(self.out)
        self.gmt_files_list.append(cpt_fle)
        if logscale:
            self.cmds.append("gmt makecpt -Cjet -T{}/{}/30+n -Q -D > \
                             {}".format(min(data), max(data), cpt_fle))
        else:
            self.cmds.append("gmt makecpt -Cjet -T{}/{}/30+n -D > \
                             {}".format(min(data), max(data), cpt_fle))


        df = pd.DataFrame({'lo':longitude, 'la':latitude, 'c':data})
        dat_tmp = '{}/tmp_dat_col{}.csv'.format(self.out, label.replace(' ','-'))
        self.gmt_files_list.append(dat_tmp)
        df.sort_values(by=['c']).to_csv(dat_tmp, index = False, header = False)

        space = np.floor(abs(min(data)-max(data))/3)
        self.cmds.append('gmt plot {} {}{} -C{}'.format(dat_tmp, shape, size, cpt_fle))
        self.cmds.append('gmt colorbar -DJBC -Ba{}+l{} -C{}'.format(space, label, cpt_fle))

    def add_size_scaled_points(self, longitude, latitude, data, shape='-Ss',
            logplot=False, color='blue', smin=0.01, coeff=1.0, sscale=2.0, label='',
            legend=True):
        '''
        Adds xy data (epicenters) size-scaled by some specified data value
        :param array longitude:
            x coordinates/longitudes of data to be plotted
        :param array latitude:
            y coordinates/latitudes of data to be plotted
        :param array data:
            array to be used to size-scale the xy data
        :param str shape: 
            shape of the plotted data. Must start with '-S'. Default is a square.
            See GMT documentation.
            https://docs.generic-mapping-tools.org/latest/psxy.html#s
        :param float size:
            size of the plotted symbols
        :param logplot:
            if True, scale colors in log space
        :param str color:
            color of the plotted symbols
        :param float smin:
            sets size of the smallest symbol 
        :param float coeff:
            with sscale, sets relative size among data values
        :param float sscale:
            with coeff, sets relative size among data values
            set sscale=None to use constant size set by coeff
        :param str label:
            Data label for the legend. Also used to name tmp file
        '''

        if logplot:
            data = np.log10(data.copy())

        if sscale is None:
            sz = [coeff] * len(latitude)
        else: 
            sz = smin + coeff * data ** sscale

        df = pd.DataFrame({'lo':longitude, 'la':latitude, 's':sz})
        dat_tmp = '{}/tmp_dat_size{}.csv'.format(self.out, label.replace(' ','-'))
        self.gmt_files_list.append(dat_tmp)
        df.to_csv(dat_tmp, index = False, header = False)


        self.cmds.append('gmt plot {} {} -G{} -Wblack'.format(dat_tmp, shape, color))

        mindat = np.floor(min(data))
        maxdat = np.ceil(max(data))
        drange = abs(mindat - maxdat)
        
        ds = np.arange(mindat,maxdat+1,np.ceil(drange/5))

        if legend:
            self._add_legend_size_scaled(ds, color, sz, shape, label, sscale)

    def _add_legend_size_scaled(self, data, color, size, shape, label, sscale):
        '''
        adds legend for catalogue seismicity.  
        '''

        fname = '{}/legend_ss.csv'.format(self.out)
        chk_file = 1 if os.path.isfile(fname) else 0

        
        if chk_file == 0:
            self.gmt_files_list.append(fname)
            fou = open(fname, 'w')
            if sscale is not None:
                fou.write("L 12p R {}\n".format(label))
                fou.write('G 0.1i\n')
        else:
            fou = open(fname, 'a')

        if sscale is not None:
            fmt = "S 0.4i {} {:.4f} {} 0.0c,black 2.0c {:.0f} \n"
    
            sh = shape.replace('-S','').replace("'",'')
    
            for dd,ss in zip(data, size):
                fou.write(fmt.format(sh, ss, color, dd))
                fou.write('G 0.2i\n')
    
        else:
            fou = open(fname,'a')
            fmt = "S 0.4i {} {} {} 0.0c,black 2.0c {} \n"
            sh = shape.replace('-S','').replace("'",'')
            fou.write(fmt.format(sh, size[0], color, label))

        fou.close()
    
        tmp = "gmt legend {} -DJMR -C0.3c ".format(fname)
        tmp += "--FONT_ANNOT_PRIMARY=12p"

        chk = sum([1 if c.find(fname)>0 else 0 for c in self.cmds])
        add_plot_line = 0 if chk > 0 else 1

        if add_plot_line==1:
            self.cmds.append(tmp)

    def _select_color_mag(self, mag):
        '''
        sets colors to magntidues - currently not in use anywhere
        '''
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
        :param string filename:
            the filename containing the gcmt entries
        :param string mech_format: 
            the format of the file to be plotted. 
            https://docs.generic-mapping-tools.org/latest/supplements/seis/psmeca.html?highlight=psmeca#s
            currently only focal mechanism (mech_format='FM') and seimsic 
            moment tensor (mech_format='MT') are supported, both using the 
            Harvard CMT convention
        '''

        if mech_format == 'FM':
            mf = 'c'
        elif mech_format == 'MT':
            mf = 'm'
        else:
            fail_error = "mech_format must be either 'FM' or 'MT'; see doc"
            raise ValueError(fail_error)

        if config is not None:
            #df = pd.read_csv(filename)
            # TODO: make some other settings... scale by mag, color, don't
            # use label, etc
            print('config methods not implemented yet')
        else:
            self.cmds.append("gmt psmeca {} -S{}0.5 -t20".format(filename, mf))


    def add_catalogue_cluster(self):
        #TODO
        pass

    def savemap(self, filename=None, save_script=False, verb=False):
        '''
        Saves map by finalizing GMT script and executing it line by line
        :param string filename:
            filename for output. include the suffix which indicates the desired 
            file type. limited to "pdf", "png", and "jpg". defaults to pdf
        :param boolean save_script:
            true in order to save the GMT script and its needed files. 
            if false, the temporary files are erased
        :param verb:
            if True, print GMT commands during execution
        '''

        # set file name and file type 
        if filename != None and "." in filename[-4:]:
            filetype = filename[-3:]
            filestring = filename[:-4]
            
        elif filename != None and "." not in filename[-4:]:
            filetype = "pdf"
            filestring = filename
        else:
            filetype = "pdf"
            filestring = "map"

        # remove any old instances of gmt end, then re-add
        # necessary in case plotting occurs at differt stages

        self.cmds=[x for x in self.cmds if x != "gmt end" and "gmt figure" not in x]

        self.cmds.insert(1, "gmt figure {}/{} {}".format(self.out, filestring, filetype))
        self.cmds.append("gmt end")

        self._check_output(filename)

        for cmd in self.cmds:
            if verb:
                print(cmd)
            out = subprocess.call(cmd, shell=True)

        print("Map saved to {}/{}.{}.".format(self.out, filestring, filetype))
        
        if save_script==True:
            self._save_gmt_script(scriptname=filename.replace(filetype,'sh'))
        else:
            [os.remove(fi) for fi in self.gmt_files_list]

    def _save_gmt_script(self, scriptname="gmt_plotter.sh"):
        '''
        saves the gmt plotting commands as a shell script
        :param string filename:
            filename to use for saved GMT script 
        '''

        if self.cmds[-1] != "gmt end":
            self.cmds.append("gmt end")

        fname = '{}/{}'.format(self.out, scriptname)
        
        with open(fname,'w') as f:
            f.write('\n'.join(self.cmds))

        print("GMT script written to {}.".format(fname))

    def show(self):
        '''
        Show the pdf in ipython
        '''
        #TO DO
        pass

