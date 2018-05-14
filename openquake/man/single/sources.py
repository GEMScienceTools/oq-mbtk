import numpy


def get_discrete_mfds(model):
    """
    Get discrete MFDs

    :parameter model:
        A list of hazardlib source instances
    :returns:
        A list of tuples where each tuple is a MFD
    """
    mfds = []
    for src in model:
        out = src.mfd.get_annual_occurrence_rates()
        mfds.append(numpy.array(out))
    return mfds
