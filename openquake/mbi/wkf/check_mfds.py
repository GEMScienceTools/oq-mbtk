#!/usr/bin/env python
# coding: utf-8

from openquake.baselib import sap
from openquake.wkf.mfd import check_mfds


def main(fname_input_pattern, fname_config, *, src_id=None):
    check_mfds(fname_input_pattern, fname_config, src_id)


main.fname_input_pattern = "Pattern to as set of OQ Engine .xml files with a SSM"
main.fname_config = "Name of the configuration file"
main.src_id = "The ID of the source to use"

if __name__ == '__main__':
    sap.run(check_mfds)
