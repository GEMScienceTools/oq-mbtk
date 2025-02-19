import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from glob import glob
from shapely import Polygon as ShapelyPolygon, LineString

from openquake.commonlib import readinput
from openquake.hazardlib.nrml import GeometryModel
from openquake.hazardlib.geo.mesh import RectangularMesh
from openquake.hazardlib.geo.surface.simple_fault import SimpleFaultSurface
from openquake.hazardlib.source.multi_fault import MultiFaultSource
from openquake.hazardlib.source.complex_fault import ComplexFaultSource
from openquake.hazardlib.source.simple_fault import SimpleFaultSource
from openquake.hazardlib.source.kite_fault import KiteFaultSource
from openquake.hazardlib.source.characteristic import CharacteristicFaultSource
from openquake.hazardlib.source.complex_fault import ComplexFaultSurface
from openquake.hazardlib.source.point import PointSource


def plot_mfd_cumulative(mfd, fig=None, label='', color=None, linewidth=1,
                        title=''):
    aa = np.array(mfd.get_annual_occurrence_rates())
    cml = np.cumsum(aa[::-1, 1])
    if color is None:
        color = np.random.rand(3)
    plt.plot(aa[:, 0], cml[::-1], label=label, lw=linewidth, color=color)
    plt.title(title)


def plot_mfd(mfd, fig=None, label='', color=None, linewidth=1):
    bw = 0.1
    aa = np.array(mfd.get_annual_occurrence_rates())
    occs = []
    if color is None:
        color = np.random.rand(3)
    for mag, occ in mfd.get_annual_occurrence_rates():
        plt.plot([mag-bw/2, mag+bw/2], [occ, occ], lw=2, color='grey')
        occs.append(occ)
    plt.plot(aa[:, 0], aa[:, 1], label=label, lw=linewidth)


def plot_models(models):
    """
    :param models:
    """
    fig = plt.figure()
    # MN: 'ax' assigned but never used
    ax = fig.add_subplot(111)
    for model in models:
        for src in model:
            plot_source(src)


def plot_source(src):
    """
    :param src:
    """
    if isinstance(src, PointSource):
        plt.plot(src.location.longitude, src.location.latitude, 'sg')
    else:
        print(type(src))
        raise ValueError('Unhandled exception')


def plot_polygon(poly):
    """
    :param src:
    """
    plt.plot(poly.lons, poly.lats, '-b', linewidth=3)


def plot_end():
    plt.show()


def get_ssm_files(model_dir):
    """
    For a given source model get the xml ssm files
    """
    # Get seismic source model XMLs
    base_path = os.path.join(model_dir, 'in', 'ssm')
    try:
        assert os.path.exists(base_path)
    except:
        raise ValueError("Admitted model does not abide to required directory "
                         "hierarchy: model_dir/in/ssm/")
    files = []

    # Loop through each subfolder and get all XML files
    for depth in range(5):  # Search up to 5 levels deep
        fipath = os.path.join(base_path, *['**']*depth)
        files.extend(glob(fipath + "/*.xml", recursive=True))

    # Remove duplicates from recursive search
    files = pd.Series(files).drop_duplicates().values

    return files


def get_sources(model_dir, inv_time, rms):
    """
    Load the sources in the given model and return them as a list.
    """            
    # Get source XMLs
    files = get_ssm_files(model_dir)

    # Fault classess
    faults = [MultiFaultSource,
              SimpleFaultSource,
              ComplexFaultSource,
              CharacteristicFaultSource,
              KiteFaultSource]

    # Read the XMLs for all srcs in the given model
    ssm = readinput.read_source_models(files, 'tmp.hdf5', 
                                       investigation_time=inv_time,
                                       rupture_mesh_spacing=rms,
                                       area_source_discretization=5,
                                       width_of_mfd_bin=0.1)
    
    # Get a list of the source objects
    srcs = []
    geom_models = []
    for ssm_obj in ssm:
        if isinstance(ssm_obj, GeometryModel):
            geom_models.append(ssm_obj)
        else:
            for idx_sg, _ in enumerate(ssm_obj.src_groups):
                for src in ssm_obj[idx_sg]:
                    if type(src) in faults: # Only retain the fault sources
                        srcs.append(src)

    # If no MultiFaultSources set to None
    if len(geom_models) < 1:
        geom_models = None

    return srcs, geom_models


def get_boundary_2d(smsh):
    """
    Returns a polygon for the given surface.
    """
    coo = []

    # Upper boundary + trace
    idx = np.where(np.isfinite(smsh.lons[0, :]))[0]
    tmp = [(smsh.lons[0, i], smsh.lats[0, i]) for i in idx]
    trace = LineString(tmp)
    coo.extend(tmp)

    # Right boundary
    idx = np.where(np.isfinite(smsh.lons[:, -1]))[0]
    tmp = [(smsh.lons[i, -1], smsh.lats[i, -1]) for i in idx]
    coo.extend(tmp)
    
    # Lower boundary
    idx = np.where(np.isfinite(smsh.lons[-1, :]))[0]
    tmp = [(smsh.lons[-1, i], smsh.lats[-1, i]) for i in np.flip(idx)]
    coo.extend(tmp)
    
    # Left boundary
    idx = idx = np.where(np.isfinite(smsh.lons[:, 0]))[0]
    tmp = [(smsh.lons[i, 0], smsh.lats[i, 0]) for i in np.flip(idx)]
    coo.extend(tmp)

    return trace, ShapelyPolygon(coo)


def get_complex_mesh(src):
    """
    Get the mesh of a ComplexFaultSource
    """
    # Get the surface
    sfc = ComplexFaultSurface.from_fault_data(
        src.edges, mesh_spacing=src.rupture_mesh_spacing)
    
    # Get mesh
    mesh = RectangularMesh(sfc.mesh.lons, sfc.mesh.lats, sfc.mesh.depths)
    
    return mesh


def get_characteristic_mesh(src):
    """
    Get the mesh of a CharacteristicFaultSource
    """
    lons = src.surface.mesh.lons
    lats = src.surface.mesh.lats
    deps = src.surface.mesh.depths
    if lons.ndim == 2:
        mesh = RectangularMesh(lons, lats, deps)
    else:
        # Some char fault meshes don't have coo
        # with ndim == 2 by def (assertion error)
        mesh = RectangularMesh(np.array([lons]),
                               np.array([lats]),
                               np.array([deps]))

    return mesh
    

def get_simple_mesh(src):
    """
    Get the mesh of a SimpleFaultSource
    """
    # Get the surface
    sfc = SimpleFaultSurface.from_fault_data(
       src.fault_trace, src.upper_seismogenic_depth,
       src.lower_seismogenic_depth, src.dip,
       mesh_spacing=src.rupture_mesh_spacing)
    
    # Get mesh
    mesh = RectangularMesh(sfc.mesh.lons, sfc.mesh.lats, sfc.mesh.depths)
    
    return mesh


def get_kite_mesh(src):
    """
    Get the mesh of a KiteFaultSource
    """
    # Get the sfc (idl set to false as not automated yet in engine)
    sfc = src.surface.from_profiles(src.profiles, src.profiles_sampling,
                                    src.rupture_mesh_spacing, idl=False,
                                    align=False)
    
    # Get mesh
    mesh = RectangularMesh(sfc.mesh.lons, sfc.mesh.lats, sfc.mesh.depths)

    return mesh


def get_geoms(srcs, geom_models):
    """
    Extract the geometry of each fault source and write to a geoJSON
    """
    traces = []
    polys = []
    suids = []

    # First non-MFS sources
    for i, src in enumerate(srcs):
        if isinstance(src, MultiFaultSource): # Skip MFS as use the geom model
            continue                          # object to get these geometries
        if isinstance(src, ComplexFaultSource):
            surf = get_complex_mesh(src)
        elif isinstance(src, CharacteristicFaultSource):
            surf = get_characteristic_mesh(src)
        elif isinstance(src, SimpleFaultSource):
            surf = get_simple_mesh(src)
        elif isinstance(src, KiteFaultSource):
            surf = get_kite_mesh(src)
        else:
            raise ValueError(f"Unknown source typology admitted: ({type(src)})")
        trace, poly = get_boundary_2d(surf)        
        traces.append(trace)
        polys.append(poly)
        suids.append(i)
    
    # Get geometries of the MFS sources too
    if geom_models:
        for gm in geom_models:
            for i, key in enumerate(gm.sections):
                surf = gm.sections[key]
                trace, poly = get_boundary_2d(surf.mesh)
                traces.append(trace)
                polys.append(poly)
                suids.append(i)

    daf = pd.DataFrame({'suid': suids, 'geometry': polys})
    gdaf_polys = gpd.GeoDataFrame(daf, geometry='geometry') 
    daf = pd.DataFrame({'suid': suids, 'geometry': traces})
    gdaf_traces = gpd.GeoDataFrame(daf, geometry='geometry') 

    return gdaf_polys, gdaf_traces


def export_faults(gdaf_polys, gdaf_traces, model_dir):
    """
    Export the GeoDataFrames of the fault traces and fault polygons
    """
    out_traces = os.path.join(model_dir, 'fault_traces.geojson')
    out_polys = os.path.join(model_dir, 'fault_sections.geojson')
    gdaf_traces.to_file(out_traces)
    gdaf_polys.to_file(out_polys)
    print(f"Fault traces exported to {out_traces}")
    print(f"Fault sections exported to {out_polys}")

    return out_polys, out_traces 


def plot_faults(gdaf_polys, gdaf_traces, region, model_dir):
    """
    Plot the faults within the geoJSONs
    """
    import pygmt 
    out = os.path.join(model_dir, "faults_plot.png")
    fig = pygmt.Figure()
    fig.basemap(region=region, projection="M15c", frame=True)
    fig.coast(land="grey", water="skyblue")
    fig.plot(gdaf_polys, pen="0.5p,red")
    fig.plot(gdaf_traces, pen="1p,green")
    fig.savefig(out)
    print(f"Faults plotted and saved to {out}")
    fig.show()


def get_fault_geojsons(model_dir, inv_time, rms, plotting=False,
                       plotting_region=None):
    """
    Write the fault sections and fault traces within the given hazard model
    model to geojsons.

    :param model_dir: directory containing the required hazard model

    :param inv_time: Investigation time to use when parsing the SSC (for
                     non-parametric sources)

    :param rms: Rupture mesh spacing to use when parsing the SSC

    :param plotting: Boolean which if True creates a plot using the geoJSONs
                     of the faults in the given hazard model

    :param plotting_region: list of [min lon, max lon, min lat, max lat] used
                            to define axis limits of the plotted geoJSONs
    """
    # Get the sources in the given model
    srcs, geom_models = get_sources(model_dir, inv_time, rms)

    # Now get the geometries
    gdaf_polys, gdaf_traces = get_geoms(srcs, geom_models)

    # Export into geoJSONs
    out_polys, out_traces = export_faults(gdaf_polys, gdaf_traces, model_dir)

    # Plot them if specified
    if plotting:
        plot_faults(gdaf_polys, gdaf_traces, plotting_region, model_dir)

    return gdaf_polys, gdaf_traces, out_polys, out_traces