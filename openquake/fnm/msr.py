#!/usr/bin/env python
# coding: utf-8

import numpy as np

def area_to_mag(area, type='generic'):
    if type == 'generic':
        return np.log10(area) + 4.0
    else:
        raise ValueError("MSR not supported")
