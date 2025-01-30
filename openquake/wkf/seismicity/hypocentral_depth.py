
import numpy as np
import pandas as pd
from typing import Tuple
import matplotlib.pyplot as plt


def hypocentral_depth_analysis(
        fname: str, depth_min: float, depth_max: float, depth_binw: float,
        figure_name_out: str = '', show: bool = False, depth_bins=[], remove_fixed=[], label='',
        figure_format='png') -> Tuple[np.ndarray, np.ndarray]:
    """
    :param fname:
        The name of the file containing the catalogue
    :param depth_min:
        The minimum depth [km]
    :param depth_max:
        The maximum depth [km]
    :param depth_binw:
        The width of the bins used [km]. Alternatively it's possible to use
        the bins by setting the `bins` variable.
    :param figure_name_out:
        The name of the figure to be created
    :param show:
        When true the show figures on screen
    :param depth_bins:
        The bins used to build the statistics. Overrides the `depth_min`,
        `depth_max`, `depth_binw` combination.
    :param remove_fixed:
    	list of fixed depth values to be removed from the analysis
    :param label:
        A label used in the title of the figure
    :param figure_format:
        Format of the figure
    """

    # Read the file as a pandas Dataframe
    df = pd.read_csv(fname)

    if len(df.depth) < 1:
        return None, None

    # Set depth intervals
    if len(depth_bins) < 1:
        bins = np.arange(depth_min, depth_max+depth_binw*0.1, depth_binw)
    else:
        bins = np.array([float(a) for a in depth_bins])
        depth_max = max(bins)
        depth_min = min(bins)

    # Filter the catalogue
    df = df[(df.depth > depth_min) & (df.depth <= depth_max)]
    
    # remove_fixed removes fixed depths from the analysis
    # This redistributes the pdf omitting the fixed depth events
    if len(remove_fixed) > 0:
        df = df[~df.depth.isin(remove_fixed)]    

    # Build the histogram
    hist, _ = np.histogram(df['depth'], bins=bins)

    if show or len(figure_name_out):

        # Create the figure
        fig, ax1 = plt.subplots(constrained_layout=True)
        heights = np.diff(bins)

        plt.barh(bins[:-1], width=hist, height=heights, align='edge',
                 hatch='///', fc='none', ec='blue', alpha=0.5)

        ax1.set_ylim([depth_max, depth_min])
        ax1.invert_yaxis()
        ax1.grid(which='both')
        ax1.set_xlabel('Count')
        ax1.set_ylabel('Depth [km]')

        ax2 = ax1.twiny()
        ax2.invert_yaxis()
        ax2.set_ylim([depth_max, depth_min])
        ax2.set_xlim([0, 1.0])
        color = 'tab:red'
        ax2.set_xlabel('Normalized count', color=color)
        ax2.tick_params(axis='x', labelcolor=color)

        plt.barh(bins[:-1], width=hist/sum(hist), height=heights, color='none',
                 edgecolor=color, linewidth=2.0, align='edge')

        # PMF labels
        import matplotlib.patheffects as pe
        path_effects = [pe.withStroke(linewidth=4, foreground="lightgrey")]
        for x, y in zip(hist/sum(hist), bins[:-1]+depth_binw*0.5):
            ax2.text(x, y, "{:.2f}".format(x), path_effects=path_effects)

        # Set the figure title
        ax2.set_title('Source: {:s}'.format(label), loc='left')

        # Save the figure (if requested)
        if len(figure_name_out):
            plt.savefig(figure_name_out, format=figure_format)

        # Show the figure (if requested)
        if show:
            plt.show()
        plt.close()

    return hist, bins
