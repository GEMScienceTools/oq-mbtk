import os
import pandas as pd
import numpy as np
from glob import glob
from openquake.baselib import sap
from matplotlib import pyplot as plt
from openquake.cat.completeness.analysis import read_compl_data, _make_ctab
from openquake.hazardlib.mfd import TruncatedGRMFD
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse

def _read_results(results_dir):
    fils = glob(os.path.join(results_dir, 'full*'))
    print(f'Total instances: { len(fils)}')
    #    avals, bvals, weis = [], [], []
    dfs = []
    for ii, fi in enumerate(fils):
        if ii%50==0:
            print(f'reading instance {ii}')
        df = pd.read_csv(fi)
        idx = fi.split('_')[-1].replace('.csv', '')
        df['idx'] = [idx] * len(df)
        dfs.append(df)
    
    df_all = pd.concat(dfs, ignore_index=True)

    mags, rates = [], []
    cm_rates = []
    for ii in range(len(df_all)):
        row = df_all.iloc[ii]
        mags.append([float(m) for m in row.mags[1:-1].split(', ')])
        rate = [float(m) for m in row.rates[1:-1].split(', ')]
        rates.append(rate)
        cm_rates.append([sum(rate[ii:]) for ii in range(len(rate))])

    df_all['mags'] = mags
    df_all['rates'] = rates 
    df_all['cm_rates'] = cm_rates


    return df_all

def _get_best_results(df_all):

    dfs = []
    for ii, idx in enumerate(set(df_all.idx)):
        df_sub = df_all[df_all.idx == idx]
        df_subB = df_sub[df_sub.norm == min(df_sub.norm)]
        
        dfs.append(df_subB)

    df_best = pd.concat(dfs, ignore_index=True)

    return df_best

def _make_a_b_histos(df_all, df_best, figsdir):
    fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(10,5))
    binsA = np.arange(min(df_all.agr), max(df_all.agr), 0.1)
    binsB = np.arange(min(df_all.bgr), max(df_all.bgr), 0.02)
    num_cats = len(set(df_all.idx))
    
    color = 'tab:grey'
    ax[0].set_xlabel('a-value')
    ax[0].set_ylabel('Count, all', color=color)
    ax[0].tick_params(axis='y', labelcolor=color)
    
    ax[1].set_xlabel('b-value')
    ax[1].set_ylabel('Count, all', color=color)
    ax[1].tick_params(axis='y', labelcolor=color)
    
    for ii, idx in enumerate(set(df_all.idx)):

        df_sub = df_all[df_all.idx == idx]
        if num_cats < 10:
            alpha = 0.1
        else: 
            alpha = 10/num_cats

        ax[0].hist(df_sub.agr, bins=binsA, color='gray', alpha=alpha)
        ax[1].hist(df_sub.bgr, bins=binsB, color='gray', alpha=alpha)
    ax2a = ax[0].twinx()
    ax2b = ax[1].twinx()
    ax2a.hist(df_best.agr, bins=binsA, color='red', alpha=0.2)
    ax2b.hist(df_best.bgr, bins=binsB, color='red', alpha=0.2)
    
    color = 'tab:red'
    ax2a.set_ylabel('Count, best', color=color)
    ax2a.tick_params(axis='y', labelcolor=color)
    ax2b.set_ylabel('Count, best', color=color)
    ax2b.tick_params(axis='y', labelcolor=color)
    
    figname = os.path.join(figsdir, 'a-b-histos.png')
    fig.savefig(figname, dpi=300)
    plt.close(fig)

def plt_compl_tables(compdir, figdir, df_best): 
    ctabids = df_best.id.values
    compl_tables, mags_chk, years_chk = read_compl_data(compdir)

    yrs, mgs = [],[]
    for cid in ctabids:
        ctab = _make_ctab(compl_tables['perms'][int(cid)], 
                                 years_chk, mags_chk)

        # add first
        plt.plot(ctab[0][0], ctab[0][1], 'ko', alpha=0.03)
        plt.plot([ctab[0][0], ctab[0][0]+10], [ctab[0][1], ctab[0][1]], 'r--', alpha=0.03)
        yrs.append(ctab[0][0])
        mgs.append(ctab[0][1])
    
        for ii in range(len(ctab)-1):
            plt.plot([ctab[ii][0], ctab[ii+1][0]], [ctab[ii+1][1], ctab[ii+1][1]], 'r', alpha=0.03)
            plt.plot([ctab[ii][0], ctab[ii][0]], [ctab[ii][1], ctab[ii+1][1]], 'r', alpha=0.03)
            plt.plot(ctab[ii+1][0], ctab[ii+1][1], 'ko', alpha=0.03)
    
            yrs.append(ctab[ii+1][0])
            mgs.append(ctab[ii+1][1])

    plt.title('Completeness tables: Best results/catalogue')
    plt.xlabel('Year')
    plt.ylabel('Lower magnitude threshold')
    fout1 = os.path.join(figdir, 'completeness_v1.png')
    plt.savefig(fout1, dpi=300)
    plt.close()

    plt.hist2d(yrs, mgs)
    plt.colorbar(label='Count')
    fout2 = os.path.join(figdir, 'completeness_v2.png')
    plt.savefig(fout2, dpi=300)
    plt.close()

def plt_a_b_density(df, figsdir, figname, weights=None, density=True):
    plt.hist2d(df.agr, df.bgr, bins=(10,10), weights=weights)
    plt.colorbar(label='Count')
    fout = os.path.join(figsdir, figname)
    plt.savefig(fout, dpi=300)
    plt.close()

def get_top_percent(df_all, fraction):
    min_norm = min(df_all.norm)
    max_norm = max(df_all.norm)
    thresh = abs(max_norm - min_norm) * fraction 
    df_thresh = df_all[df_all.norm <= min_norm + thresh]
     
    return df_thresh


def plot_best_mfds(df_best, figsdir):
    num = len(df_best)
    for ii in range(len(df_best)):
        row = df_best.iloc[ii]
        mfd = TruncatedGRMFD(4, 8.5, 0.2, df_best.agr.iloc[ii], df_best.bgr.iloc[ii])
        mgrts = mfd.get_annual_occurrence_rates()
        mfd_m = [m[0] for m in mgrts]
        mfd_r = [m[1] for m in mgrts]
        mfd_cr = [sum(mfd_r[ii:]) for ii in range(len(mfd_r))]
        if ii == 0:
            plt.scatter(row.mags, row.rates, marker='_', color='r', 
                        label='Incremental occurrence')
            plt.scatter(row.mags, row.cm_rates, marker='.', color='b', 
                        label='Cumulative occurrence')
            plt.semilogy(mfd_m, mfd_r, color='r', linewidth=0.1, 
                         zorder=0, label='Incremental MFD')
            plt.semilogy(mfd_m, mfd_cr, color='b',
                         linewidth=0.1, zorder=0, label='Cumulative MFD')

        else: 
            if num <= 10:
                alpha1 = 0.1
            else:
                alpha1 = 10/num

            plt.scatter(row.mags, row.rates, marker='_', color='r', 
                        alpha=alpha1)
            plt.scatter(row.mags, row.cm_rates, marker='.', color='b', 
                        alpha=alpha1)
            plt.semilogy(mfd_m, mfd_r, color='r', alpha=3*alpha1, linewidth=0.1, 
                         zorder=0)
            plt.semilogy(mfd_m, mfd_cr, color='b', alpha=5*alpha1, 
                         linewidth=0.1, zorder=0)

    plt.xlabel('Magnitude')
    plt.ylabel('Annual occurrence rates')
    plt.legend()
    fout = os.path.join(figsdir, 'mfds_best.png')
    plt.savefig(fout, dpi=300)
    plt.close()

# Function to calculate normalized histogram for a group
def norm_histo(group, field='rates', bins=10):
    counts, bin_edges = np.histogram(group[field], bins=bins, density=True)
    # Normalize counts to ensure the area under the histogram equals 1
    bin_widths = np.diff(bin_edges)
    normalized_counts = counts * bin_widths
    bin_centers = [0.5*(bin_edges[ii] + bin_edges[ii+1]) for ii in range(len(bin_edges)-1)]
    alpha = (normalized_counts - min(normalized_counts)) / (max(normalized_counts) - min(normalized_counts))

    return bin_centers, alpha

def weighted_mean(values, weights):
    return np.sum(values * weights) / np.sum(weights)

def weighted_covariance(x, y, weights):
    mean_x = weighted_mean(x, weights)
    mean_y = weighted_mean(y, weights)
    cov_xy = np.sum(weights * (x - mean_x) * (y - mean_y)) / np.sum(weights)
    cov_xx = np.sum(weights * (x - mean_x)**2) / np.sum(weights)
    cov_yy = np.sum(weights * (y - mean_y)**2) / np.sum(weights)
    return np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])


def plot_weighted_covariance_ellipse(df, figdir, n_std=1.0, 
                                     figname='a-b-covariance.png'):

    # set up data
    x = df.agr
    y = df.bgr
    wei = 1-df.norm
    weights = (wei - min(wei)) / (max(wei) - min(wei)) 

    # set up plot
    fig, ax = plt.subplots()
    hb = ax.hexbin(x, y, gridsize=20, cmap='Blues')
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('counts')

    # get covariance
    cov_matrix = weighted_covariance(x, y, weights)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort the eigenvalues and eigenvectors
    order = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    
    # Get the index of the largest eigenvalue
    largest_eigvec = eigenvectors[:, 0]
    
    angle = np.degrees(np.arctan2(largest_eigvec[1], largest_eigvec[0]))
    width, height = 2 * n_std * np.sqrt(eigenvalues)
    
    ellipse = Ellipse(xy=(weighted_mean(x, weights), weighted_mean(y, weights)),
                      width=width, height=height, angle=angle,
                      facecolor='none', edgecolor='red')
    
    ax.add_patch(ellipse)

    angle_rad = np.radians(angle)  # angle computed during the ellipse plotting

    center_x = weighted_mean(x, weights)
    center_y = weighted_mean(y, weights)

    a = np.sqrt(eigenvalues[0])  # Length of semi-major axis
    b = np.sqrt(eigenvalues[1])  # Length of semi-minor axis

    # For the semi-major axis (a)
    major_x1 = center_x + a * np.cos(angle_rad)
    major_y1 = center_y + a * np.sin(angle_rad)
    
    major_x2 = center_x - a * np.cos(angle_rad)
    major_y2 = center_y - a * np.sin(angle_rad)
    
      
    ax.scatter(center_x, center_y, c='white', marker='s', edgecolors='red')
    ax.scatter(major_x1, major_y1, c='white', marker='o', edgecolors='red')
    ax.scatter(major_x2, major_y2, c='white', marker='o', edgecolors='red')

    color = 'red'
    ax.text(0.02, 0.98, f'a = {np.round(major_x1, 3)}, b = {np.round(major_y1, 3)}',
            transform=ax.transAxes, verticalalignment='top', horizontalalignment='left', 
            fontsize=12, color=color)
    ax.text(0.02, 0.94, f'a = {np.round(center_x, 3)}, b = {np.round(center_y, 3)}',
            transform=ax.transAxes, verticalalignment='top', horizontalalignment='left', 
            fontsize=12, color=color)
    ax.text(0.02, 0.90, f'a = {np.round(major_x2, 3)}, b = {np.round(major_y2, 3)}',
            transform=ax.transAxes, verticalalignment='top', horizontalalignment='left', 
            fontsize=12, color=color)
    

    ax.set_xlabel('a-value', fontsize=12)
    ax.set_ylabel('b-value', fontsize=12)
    ax.set_title(figname.replace('.png', ''))

    fout = os.path.join(figdir, figname)
    plt.savefig(fout, dpi=300)
    plt.close()

    return center_x, center_y, major_x1, major_y1, major_x2, major_y2



def plot_all_mfds(df_all, df_best, figsdir, field='rates', bins=10, bw=0.2, figname=None):
# Group the DataFrame by the 'Category' column and apply the histogram calculation function

    fl_mags = [item for sublist in df_all.mags.values for item in sublist]
    fl_rates = [item for sublist in df_all.rates.values for item in sublist]
    fl_crates = [item for sublist in df_all.cm_rates.values for item in sublist]
    fl_df = pd.DataFrame({'mags': fl_mags, 'rates': fl_rates, 'cm_rates': fl_crates})

    grouped = fl_df.groupby('mags')
    hist_data = grouped.apply(lambda g: norm_histo(g, field=field, bins=bins))
    mags = hist_data._mgr.axes[0].values
    results = hist_data.values
    
    all_alpha = []
    for mag, rat in zip(mags, results):
        m = [mag] * len(rat[0])
        nmc = rat[1]
        all_alpha.extend(nmc)
    alph_min = min(all_alpha)
    alph_max = max(all_alpha)
    
    norm = mcolors.Normalize(vmin=alph_min, vmax=alph_max)
    
    # Choose a colormap
    colormap = plt.cm.Purples
    
    for mag, rat in zip(mags, results):
        m = [mag] * len(rat[0])
        alph = rat[1] 
        alph[alph<0.1]=0.2
        plt.semilogy([m[0], m[0]], [min(rat[0]), max(rat[0])], c='gray', linewidth=1, zorder=1)
        plt.scatter(m, rat[0], 250, marker='_', c=alph, cmap=colormap, norm=norm, zorder=0)

    for index, row in df_best.iterrows():
        plt.scatter(row['mags'], row[field], 2, 'k', marker='s')
        mfd = TruncatedGRMFD(min(mags)-bw, 8.5, bw, row.agr, row.bgr)
        mgrts = mfd.get_annual_occurrence_rates()
        mfd_m = [m[0] for m in mgrts]
        mfd_r = [m[1] for m in mgrts]
        if len(df_best) <=30:
            alpha = 0.1
        else:  
            alpha=30/len(df_best)
        
        if field == 'cm_rates':
            mfd_cr = [sum(mfd_r[ii:]) for ii in range(len(mfd_r))]
            plt.semilogy(mfd_m, mfd_cr, color='maroon', linewidth=0.2, zorder=10, alpha=alpha)
        else:
            plt.semilogy(mfd_m, mfd_r, color='maroon', linewidth=0.2, zorder=10, alpha=alpha)

    if figname==None:
        figname = f'mfds_all_{field}.png'

    fout = os.path.join(figsdir, figname)


    plt.xlabel('Magnitude')
    plt.ylabel('annual occurrence rate')
    plt.title(figname.replace('.png', ''))
    plt.savefig(fout)
    plt.close()


def make_all_plots(resdir_base, compdir, figsdir_base, labels):
    agrs, bgrs, labs = [], [], []
    for label in labels:
        print(f'Running for {label}')
        resdir = os.path.join(resdir_base, label)
        figsdir = os.path.join(figsdir_base, label)
        print('getting all results')
        df_all = _read_results(resdir)
        print('getting best results')
        df_best = _get_best_results(df_all)
        print('making histograms')
        _make_a_b_histos(df_all, df_best, figsdir)
        print('plotting completeness')
        plt_compl_tables(compdir, figsdir, df_best)
        print('plotting 2d histo')
        plt_a_b_density(df_best, figsdir, 'a-b-density_best.png')
        plt_a_b_density(df_all, figsdir, 'a-b-density_all.png')
        df_thresh = get_top_percent(df_all, 0.2)
        plt_a_b_density(df_thresh, figsdir, 'a-b-density_20percent.png')
        nm = df_thresh.norm
        nm_weight = (nm- min(nm))/(max(nm)-min(nm))
        plt_a_b_density(df_thresh, figsdir, 'a-b-density_20percent_w.png', 
                        weights = nm_weight)
        print('plotting mfds')
        plot_best_mfds(df_best, figsdir)
        plot_all_mfds(df_all, df_best, figsdir, field='rates', bins=60)
        plot_all_mfds(df_all, df_best, figsdir, field='cm_rates', bins=60)
        plot_all_mfds(df_best, df_best, figsdir, field='rates', bins=60, 
                      figname='mfds_best_rates.png')
        plot_all_mfds(df_best, df_best, figsdir, field='cm_rates', bins=60, 
                      figname='mfds_best_cmrates.png')
        plot_all_mfds(df_all, df_thresh, figsdir, field='rates', bins=60, 
                      figname='mfds_thresh_rates.png')
        plot_all_mfds(df_all, df_thresh, figsdir, field='cm_rates', bins=60, 
                      figname='mfds_thresh_cmrates.png')
        print('plotting covariance')
        cx, cy, mx1, my1, mx2, my2 = plot_weighted_covariance_ellipse(df_best, figsdir)
        plot_weighted_covariance_ellipse(df_thresh, figsdir, figname=f'{label}-20percent.png')

        labs.extend([f'{label}-center', f'{label}-low', f'{label}-high'])
        agrs.extend([cx, mx1, mx2])
        bgrs.extend([cy, my1, my2])

    return labs, agrs, bgrs
