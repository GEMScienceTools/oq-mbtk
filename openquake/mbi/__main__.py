#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4

import sys
import logging

from openquake.baselib import sap
from openquake import mbi

PY_VER = sys.version_info[:3]
if PY_VER < (3, 6):
    sys.exit('Python 3.6+ is required, you are using %s', sys.executable)


def oqm():
    args = set(sys.argv[1:])
    level = logging.DEBUG if 'debug' in args else logging.INFO
    logging.basicConfig(level=level)
    sap.run(mbi, prog='oqm')


if __name__ == '__main__':
    oqm()
