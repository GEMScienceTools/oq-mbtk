from openquake.sub.create_2pt5_model import create_2pt5_model
from openquake.sub.get_profiles_from_slab2pt0 import get_profiles_geojson
from openquake.sub.build_complex_surface import build_complex_surface
from openquake.mbi.ccl import classify
import tempfile
import pathlib
import os
import configparser
import pandas as pd
import unittest

HERE = pathlib.Path(__file__).parent.resolve()


def test_geojson_classification_workflow():
    fname_geojson = HERE / "data" / "izu_slab2_css.geojson"
    fname_slab = HERE / "data" / "izu_slab2_dep_02.24.18.grd"
    classification_ini = HERE / "data" / "mariana_classification.ini"

    ## these go in a tmp folder...
    top_level = tempfile.mkdtemp()
    profiles = tempfile.mkdtemp(dir=top_level)
    sfc_out = tempfile.mkdtemp(dir=top_level)

    # edit config to find the tmp file
    config = configparser.ConfigParser()
    config.read(classification_ini)
    config.set("general", "root_folder", top_level)

    cat = HERE / "data" / "mariana_full_2202.pkl"
    config.set("general", "catalogue_filename", str(cat))

    ### This data comes from CRUST1.0, part of the Reference Earth Model
    ## (http://igppweb.ucsd.edu/~gabi/rem.html)
    ## Laske, G., Masters., G., Ma, Z. and Pasyanos, M.,
    ## Update on CRUST1.0 - A 1-degree Global Model of Earth's Crust,
    ## Geophys. Res. Abstracts, 15, Abstract EGU2013-2658, 2013.
    ## Downloaded from https://igppweb.ucsd.edu/~gabi/crust1.html#download
    crust = HERE / "data" / "depthtomoho.xyz"

    config.set("crustal", "crust_filename", str(crust))
    with open(classification_ini, "w") as configfile:
        config.write(configfile)

    max_sampl_dist = 25.0
    upper_depth_int = 0.0
    lower_depth_int = 50.0
    lower_depth_slab = 300.0

    slb = get_profiles_geojson(fname_geojson, fname_slab, spacing=10.0)
    slb.write_profiles(profiles)
    
    # Compare slab profiles
    import pandas as pd

    df1 = pd.read_csv(os.path.join(profiles, "cs_000.csv"))
    df2 = pd.read_csv(HERE / "data" / "cs_000.csv")
    
    # Compare dataframes
    pd.testing.assert_frame_equal(df1,df2, check_exact=False, atol=1e-3, rtol=0)

    create_2pt5_model(profiles, sfc_out, max_sampl_dist)

    sfc_in = os.path.join(top_level, "sfc_in")
    os.makedirs(sfc_in)

    sfc_sl = os.path.join(top_level, "sfc_sl")
    os.makedirs(sfc_sl)

    build_complex_surface(
        sfc_out, max_sampl_dist, sfc_in, upper_depth_int, lower_depth_int
    )

    build_complex_surface(
        sfc_out, max_sampl_dist, sfc_sl, lower_depth_int, lower_depth_slab
    )

    classify.classify(classification_ini, True)
    
    # Test the distances are the same 
    dists = pd.read_pickle(os.path.join(top_level, "distances", "vert_dist_to_slab_slab.pkl"))
    test_dists = pd.read_pickle(HERE / "data" / "vert_dist_to_slab_slab.pkl")
    assert len(dists) == len(test_dists)
    for a, b in zip(dists, test_dists):
        assert a == b
        
    # Test the slab depths are the same 
    depths = pd.read_pickle(os.path.join(top_level, "distances", "sub_depths_slab.pkl"))
    test_depths = pd.read_pickle(HERE / "data" / "sub_depths_slab.pkl")
    assert len(depths) == len(test_depths)
    for a, b in zip(depths, test_depths):
        assert a == b


    # Test the resulting classification is as expected
    classified = pd.read_csv(os.path.join(top_level, "classified_earthquakes.csv"))
    test_set = pd.read_csv(HERE / "data" / "classified_earthquakes_test.csv")

    assert len(classified["tr"]) == len(test_set["tr"])
    for a, b in zip(classified["tr"], test_set["tr"]):
        assert a == b
