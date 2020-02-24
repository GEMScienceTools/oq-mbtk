from typing import Callable

from osgeo import gdal
import numpy as np
from numpy.lib.stride_tricks import as_strided


def make_2d_array_strides(arr, window_radius, linear=True):
    """
    Creates an array of strides representing indices for windows/sub-arrays
    over a larger array, for rapid calculations of rolling window functions.

    :param arr:
        Array of values to be used in the calculations.

    :param window_radius:
        Radius of the window (not counting the origin) in array count units.
    
    :param linear:
        Flag specifying the shape of the stride collection.

    :returns:
        asdf

    Slightly modified from https://gist.github.com/thengineer/10024511
    """

    if np.isscalar(window_radius):
        window_radius = (window_radius, window_radius)

    ax = np.zeros(
        shape=(
            arr.shape[0] + 2 * window_radius[0],
            arr.shape[1] + 2 * window_radius[1],
        )
    )
    ax[:] = np.nan
    ax[
        window_radius[0] : ax.shape[0] - window_radius[0],
        window_radius[1] : ax.shape[1] - window_radius[1],
    ] = arr

    shape = arr.shape + (1 + 2 * window_radius[0], 1 + 2 * window_radius[1])
    strides = ax.strides + ax.strides
    s = as_strided(ax, shape=shape, strides=strides)

    return s.reshape(arr.shape + (shape[2] * shape[3],)) if linear else s


def rolling_array_operation(
    array: np.ndarray, func: Callable, window_size: int, trim: bool = False
) -> np.ndarray:
    """
    Rolls a function that operates on a square subarray over the array. The
    values in the resulting array will be at the center of each subarray.

    :param array:
        Array of values that the window function will be passed over.
        
    :param func:
        Function to be applied to each subarray. Should operate on an array and
        return a scalar.

    :param window_size:
        Dimension of the (square) sub-array or window in array counts, not in
        spatial (or other dimensional) units. Should be an odd
        integer, so that the result of the function can be unambiguously applied
        to the center of the window.

    :param trim:
        Boolean flag to trim off the borders of the resulting array, from where
        the window overhangs the array.
    """
    if window_size % 2 != 1:
        raise ValueError(
            "window_size should be an odd integer; {} passed".format(
                window_size
            )
        )

    window_rad = window_size // 2

    strides = make_2d_array_strides(array, window_rad)
    strides = np.ma.masked_array(strides, mask=np.isnan(strides))

    result = func(strides, axis=-1).data

    if trim:
        result = result[window_rad:-window_rad, window_rad:-window_rad]

    return result


def rolling_raster_operation(
    in_raster,
    func: Callable,
    window_size: int,
    outfile=None,
    raster_band: int = 1,
    trim: bool = False,
    write: bool = False,
):
    if trim == True:
        return NotImplementedError("Trimming not supported at this time.")

    if outfile is None:
        if write == True:
            raise ValueError("Must specify raster outfile")
        else:
            outfile = "./tmp.tiff"

    ds = gdal.Open(in_raster)

    rast = ds.GetRasterBand(raster_band).ReadAsArray()
    rast = np.asarray(rast)

    new_arr = rolling_array_operation(rast, func, window_size, trim=trim)

    drv = gdal.GetDriverByName("GTiff")

    new_ds = drv.Create(
        outfile,
        xsize=new_arr.shape[1],
        ysize=new_arr.shape[0],
        bands=1,
        eType=gdal.GDT_Float32,
    )

    new_ds.SetGeoTransform(ds.GetGeoTransform())
    new_ds.SetProjection(ds.GetProjection())
    new_ds.GetRasterBand(1).WriteArray(new_arr)

    if write:
        new_ds.FlushCache()
        new_ds = None

    return new_ds


def relief(arr, axis=-1):
    return np.amax(arr, axis=axis) - np.amin(arr, axis=axis)


def make_local_relief_raster(
    input_dem, window_size, outfile=None, write=False, trim=False
):
    relief_arr = rolling_raster_operation(
        input_dem, relief, window_size, outfile, write=write, trim=trim
    )

    return relief_arr
