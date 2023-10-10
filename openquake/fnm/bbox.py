#!/usr/bin/env python
# coding: utf-8

import numpy as np
from openquake.hazardlib.geo.geodetic import geodetic_distance


def get_bb_distance(bb1, bb2):
    """
    See shorturl.at/CWY57
    """
    # def rect_distance((x1, y1, x1b, y1b), (x2, y2, x2b, y2b)):
    x1 = bb1[0]
    y1 = bb1[2]
    x1b = bb1[1]
    y1b = bb1[3]

    x2 = bb2[0]
    y2 = bb2[2]
    x2b = bb2[1]
    y2b = bb2[3]

    left = x2b < x1  # Second bb is left of the first one
    right = x1b < x2  # Second bb is right of the first one
    bottom = y2b < y1  # Second bb is below  the first one
    top = y1b < y2  # Second bb is above the first one

    if top and left:
        return geodetic_distance(x1, y1b, x2b, y2)
    elif left and bottom:
        return geodetic_distance(x1, y1, x2b, y2b)
    elif bottom and right:
        return geodetic_distance(x1b, y1, x2, y2b)
    elif right and top:
        return geodetic_distance(x1b, y1b, x2, y2)
    elif left:
        # return x1 - x2b
        return geodetic_distance(x1, y1, x2b, y1)
    elif right:
        # return x2 - x1b
        return geodetic_distance(x2, y2, x1b, y2)
    elif bottom:
        # return y1 - y2b
        return geodetic_distance(x1, y1, x1, y2b)
    elif top:
        # return y2 - y1b
        return geodetic_distance(x2, y2, x2, y1b)
    else:
        # rectangles intersect
        return 0.


def get_bb_distance_matrix(bboxes: list) -> np.ndarray:
    """
    Given a list of bounding boxes, this function computes a matrix defining
    distances
    """
    # Compute distances between bounding boxes
    bb_dist_matrix = np.zeros((len(bboxes), len(bboxes)))
    for i in range(len(bboxes)):
        for j in range(i+1, len(bboxes)):
            bb_dist_matrix[i, j] = get_bb_distance(bboxes[i], bboxes[j])
    # np.set_printoptions(linewidth=110)
    # print(np.array_str(bb_dist_matrix, precision=1, suppress_small=True))
    return bb_dist_matrix
