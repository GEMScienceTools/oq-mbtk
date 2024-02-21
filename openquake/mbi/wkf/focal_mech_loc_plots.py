from matplotlib.gridspec import GridSpec
import numpy as np


def focal_mech_loc_plots(fname, figsize = (15, 10), width = 0.5, size = 0.1):
    """
    Produce a figure consisting of:
    	1) nodal planes plotted in space (lat/Lon) with Kaverina classification colours
    	2) scatterplot of event Kaverina classificatons and magnitudes
    	3) scatterplot of event strike vs rake, coloured by Kaverina classification
    Please note that the 'width' parameter might need to be adjusted for different models 
    """
    
    cmt_cat_zone = pd.read_csv(fname)
    plungeb = cmt_cat_zone['plunge_b']
    plungep = cmt_cat_zone['plunge_p']
    plunget = cmt_cat_zone['plunge_t']
    mclass = ['']*len(plunget)

    for i in range(0, len(plungeb)):
        mclass[i] = mecclass(plunget[i], plungeb[i], plungep[i])

    cmt_cat_zone['class'] = mclass    
    
    mts = np.column_stack([cmt_cat_zone.strike1, cmt_cat_zone.dip1, cmt_cat_zone.rake1])
    
    fig = plt.figure(layout="constrained", figsize = figsize)
    gs = GridSpec( 2, 3, figure=fig)
    a0 = fig.add_subplot(gs[0:, :-1])
    
    a0.set_xlim(np.min(cmt_cat_zone['longitude']) - 0.1, np.max(cmt_cat_zone['longitude'])+ 0.1)
    a0.set_ylim(np.min(cmt_cat_zone['latitude']) - 0.1, np.max(cmt_cat_zone['latitude']) + 0.1)

    a0.margins(0.05)

    idx = 0
    for i in range(0, len(plungeb)):
        bcc = beach(mts[idx],xy=(cmt_cat_zone['longitude'][idx], cmt_cat_zone['latitude'][idx]), width=width, linewidth=1, zorder=20, size=size, facecolor=KAVERINA[mclass[idx]])
        bcc.set_alpha(0.5)
        a0.add_collection(bcc)
        idx += 1
    
    a1 = fig.add_subplot(gs[0, -1])

    a1.scatter(cmt_cat_zone['class'], cmt_cat_zone['magnitude'], c=cmt_cat_zone['class'].map(KAVERINA))
    a1.set_xlabel("Kaverina classification")
    a1.set_ylabel("magnitude")
    
    a2 = fig.add_subplot(gs[1, -1])
    a2.scatter(cmt_cat_zone['strike1'], cmt_cat_zone['rake1'], c=cmt_cat_zone['class'].map(KAVERINA), s = 1, alpha = 0.5)
    a2.set_xlabel("strike")
    a2.set_ylabel("rake")

    fig.suptitle("Zone nodal plane distribution")
    plt.show()
