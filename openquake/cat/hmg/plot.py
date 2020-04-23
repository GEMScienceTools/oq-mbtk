import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib._color_data as mcds

from matplotlib.legend import Legend

COLORS = [mcds.XKCD_COLORS[k] for k in mcds.XKCD_COLORS]
random.seed(1)
random.shuffle(COLORS)


def get_hists(df, bins, agencies=None, column="magMw"):
    """
    :param df:
    :param bins:
    :param agencies:
    :param column:
    """
    #
    # Getting the list of agencies
    if not agencies:
        agencies = get_agencies(df)
    #
    # Creating the histograms
    out = []
    for key in agencies:
        mw = df[df['magAgency'] == key][column]
        hist, _ = np.histogram(mw, bins=bins)
        out.append(hist)
    return out


def get_ranges(agencies, df):
    #
    # Getting the list of agencies
    if not agencies:
        agencies = get_agencies(df)
    #
    # Computing the time interval
    out = []
    num = []
    for key in agencies:
        ylow = np.min(df[df['magAgency'] == key]['year'])
        yup = np.max(df[df['magAgency'] == key]['year'])
        num.append(len(df[df['magAgency'] == key]))
        out.append([ylow, yup])
    return out, num


def get_agencies(df):
    """
    :param df:
        A :class:`pandas.DataFrame` instance
    :return:
        A list
    """
    return list(df["magAgency"].unique())


def plot_time_ranges(df, agencies=None, fname='/tmp/tmp.pdf', **kwargs):
    """
    :param df:
        A :class:`pandas.DataFrame` instance
    :param agencies:
        A list of agencies codes
    :param fname:
        The name of the output file
    """
    if not agencies:
        agencies = get_agencies(df)

    # Plotting
    yranges, num = get_ranges(agencies, df)

    # Compute line widths
    max_wdt = 12
    min_wdt = 3
    lws = np.array(num)/max(num) * (max_wdt-min_wdt) + min_wdt

    # Plotting

    if "height" in kwargs:
        height = kwargs["height"]
    else:
        height = 8

    _ = plt.figure(figsize=(10, height))
    ax = plt.subplot(1, 1, 1)
    ax.tick_params(labelsize=14)
    plt.style.use('seaborn-ticks')
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['axes.labelsize'] = 16

    for i, key in enumerate(agencies):
        if sum(np.diff(yranges[i])) > 0:
            plt.plot(yranges[i], [i, i], COLORS[i], lw=lws[i])
            plt.text(yranges[i][0], i+0.2,  '{:d}'.format(num[i]))
        else:
            plt.plot(yranges[i][1], i, 'o', COLORS[i], lw=min_wdt)
            plt.text(yranges[i][1], i+0.2,  '{:d}'.format(num[i]))

    ax.grid(which='major', linestyle='-')
    ax.grid(which='minor', linestyle=':')
    xx = [' ']
    xx.extend(agencies)
    ax.set_yticks(range(len(agencies)))
    ax.set_yticklabels(agencies)
    #
    # Creating legend for thickness
    idx2 = np.argmin(num)
    idx1 = np.argmax(num)
    xlo = min(np.array(yranges)[:, 0])
    xup = max(np.array(yranges)[:, 0])
    xdf = xup - xlo
    fake1, = plt.plot([xlo, xlo], [0, 0], lw=max_wdt, alpha=1,
                      color=COLORS[idx1])
    fake2, = plt.plot([xlo, xlo], [0, 0], lw=min_wdt, alpha=1,
                      color=COLORS[idx2])
    labels = ['{:d}'.format(max(num)), '{:d}'.format(min(num))]
    leg = Legend(ax, [fake1, fake2], labels=labels, loc='best', frameon=True,
                 title='Number of magnitudes', fontsize='medium')
    ax.add_artist(leg)
    ax.set_xlim([xlo-xdf*0.05, xup+xdf*0.05])
    plt.xlabel('Year')


def plot_histogram(df, agencies=None, wdt=0.1, column="magMw",
                   fname='/tmp/tmp.pdf', **kwargs):
    """
    :param df:
        A :class:`pandas.DataFrame` instance
    :param agencies:
        A list of agencies codes
    :param wdt:
        A float defining the width of the bins
    :param fname:
        The name of the output file
    """

    # Filtering
    num = len(df)
    df = df[np.isfinite(df[column])]
    fmt = "Total number of events {:d}, with finite magnitude {:d}"
    print(fmt.format(len(df), num))
    #
    # Settings
    wdt = wdt
    if not agencies:
        agencies = get_agencies(df)
        print('List of agencies: ', agencies)
    #
    # Settings plottings
    plt.style.use('seaborn-ticks')
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['axes.labelsize'] = 16
    #
    # Data
    mw = df[column].values
    #
    # Creating bins and total histogram
    mmi = np.floor(min(mw)/wdt)*wdt-wdt
    mma = np.ceil(max(mw)/wdt)*wdt+wdt
    bins = np.arange(mmi, mma, step=wdt)
    hist, _ = np.histogram(mw, bins=bins)

    # Computing the histograms
    hsts = get_hists(df, bins, agencies, column=column)

    # Create Figure
    fig = plt.figure(figsize=(15, 8))
    ax = plt.subplot(1, 1, 1)
    ax.tick_params(labelsize=14)

    # Get the CCDF
    ccdf = np.array([sum(hist[i:]) for i in range(0, len(hist))])

    # Plotting bars of the total histogram
    plt.bar(bins[:-1]+wdt/2, hist, width=wdt*0.8, color='none',
            edgecolor='blue', align='center', lw=1, )
    #
    # Plotting the cumulative histogram
    bottom = np.zeros_like(hsts[0])
    for i, hst in enumerate(hsts):
        plt.bar(bins[:-1], hst, width=wdt*0.8, color=COLORS[i],
                edgecolor='none', align='edge', lw=1,
                bottom=bottom, label=agencies[i])
        bottom += hst
    #
    # Plotting the CCDF
    plt.plot(bins[1:], ccdf, color='red',
             label='Cumulative distribution (N>m)', lw=1)
    plt.yscale('log')
    plt.xlabel('Magnitude')
    plt.ylabel('Number of magnitude values')

    ax.grid(which='major', linestyle='-')
    ax.grid(which='minor', linestyle=':')

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5,
              fontsize='large')

    plt.savefig(fname)

    if "xlim" in kwargs:
        ax.set_xlim(kwargs["xlim"])

    if "ylim" in kwargs:
        ax.set_ylim(kwargs["ylim"])

    print('Created figure: {:s}'.format(fname))

    return fig, ax
