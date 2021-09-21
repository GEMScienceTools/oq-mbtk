
import matplotlib.pyplot as plt


def _plot_ctab(ctab, label='', xlim=None, ylim=None, color='red', ls='-',
               marker=''):
    """
    :param ctab:
        A :class:`np.ndarray` instance containing the completeness table
    """

    n = ctab.shape[0]
    if n > 1:
        for i in range(0, n-1):
            plt.plot([ctab[i, 0], ctab[i, 0]], [ctab[i, 1],
                     ctab[i+1, 1]], color=color, ls=ls, marker=marker)
            plt.plot([ctab[i, 0], ctab[i+1, 0]], [ctab[i+1, 1],
                     ctab[i+1, 1]], color=color, ls=ls, marker=marker)
        ylim = plt.gca().get_ylim()
        xlim = plt.gca().get_xlim()

    if xlim is None:
        xlim = [1900, 2020]

    if ylim is None:
        ylim = [4.5, 7.0]

    plt.plot([ctab[n-1, 0], ctab[n-1, 0]], [ylim[1], ctab[n-1, 1]],
             color=color, ls=ls, marker=marker)
    plt.plot([ctab[0, 0], xlim[1]], [ctab[0, 1], ctab[0, 1]],
             label=label, color=color, ls=ls, marker=marker)
