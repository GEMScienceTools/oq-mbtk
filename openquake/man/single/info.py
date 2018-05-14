from prettytable import PrettyTable


def print_trt_stats_table(model):
    stypes, msrtypes = get_trt_stats(model)
    for trt in stypes:
        print('Tectonic region: {0:s}'.format(trt))
        x = PrettyTable()
        x.add_column('Source type', list(stypes[trt].keys()))
        nums = [stypes[trt][key] for key in stypes[trt].keys()]
        x.add_column('Number of sources', nums)
        print(x)
        # MSR
        s = PrettyTable()
        s.add_column('MSR', list(msrtypes[trt].keys()))
        nums = [msrtypes[trt][key] for key in msrtypes[trt].keys()]
        s.add_column('Number of sources', nums)
        print(s)


def get_trt_stats(model):
    """
    Provide statistics about the sources included in the tectonic regions
    composing the model

    :parameter model:
        A list
    """
    stypes = {}
    msrtypes = {}
    for src in model:
        # Getting parameters
        trt = src.tectonic_region_type
        sty = type(src).__name__
        msr = type(src.magnitude_scaling_relationship).__name__
        # Source types
        if trt in stypes:
            if sty in stypes[trt]:
                stypes[trt][sty] += 1
            else:
                stypes[trt][sty] = 1
        else:
            stypes[trt] = {}
            stypes[trt][sty] = 1
        # MSR types
        if trt in msrtypes:
            if msr in msrtypes[trt]:
                msrtypes[trt][msr] += 1
            else:
                msrtypes[trt][msr] = 1
        else:
            msrtypes[trt] = {}
            msrtypes[trt][msr] = 1

    return stypes, msrtypes
