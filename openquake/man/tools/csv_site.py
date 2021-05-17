import re


def read_amplification_function_file(fname):
    """
    :param fname:
        Name of the .csv file
    :returns:
        A tuple with
    """

    pass


def _get_header(tmpstr):
    """
    """
    tmpstr = re.sub("^#", "", tmpstr)
    tmpstr = re.sub(",{2,", "", tmpstr)
    print(tmpstr)
