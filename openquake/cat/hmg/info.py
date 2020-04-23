from prettytable import PrettyTable


def get_table(work, mthresh=5.0, nthresh=1):
    """
    :param work:
        A :class:`pandas.DataFrame` instance
    :param mthresh:
        Minimum magnitude (original scale) threshold
    :param nthresh:
        Number of values threshold
    :return:
        An instance of :class:`prettytable.PrettyTable` and a list of list
        that can be used with the package tabulate.
    """

    x = PrettyTable()
    tbl = []
    x.field_names = ["Agency", "Magnitude Type", "Number of events"]
    other = 0
    total = 0

    prev = ""
    yyy = work[(work["value"] > mthresh)].groupby(["magAgency", "magType"])
    for name in yyy.groups:
        if len(yyy.groups[name]) > nthresh:
            total += len(yyy.groups[name])
            tmps = "" if prev == name[0] else name[0]
            prev = name[0]
            tmp = [tmps, name[1], len(yyy.groups[name])]
            x.add_row(tmp)
            tbl.append(tmp)
        else:
            other += len(yyy.groups[name])
    x.add_row(["TOTAL", "", "{:d}".format(total)])

    return x, tbl
