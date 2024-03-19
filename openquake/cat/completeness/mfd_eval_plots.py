import os
import pandas as pd
import numpy as np
from glob import glob
from openquake.baselib import sap
from matplotlib import pyplot as plt
from openquake.cat.completeness.analysis import read_compl_data, _make_ctab

def _read_results(results_dir):
    fils = glob(os.path.join(results_dir, 'full*'))
    
    #    avals, bvals, weis = [], [], []
    for ii, fi in enumerate(fils):
        df = pd.read_csv(fi)
        idx = fi.split('_')[-1].replace('.csv', '')
        df['idx'] = [idx] * len(df)
        if ii==0:
            df_all = df
        else:
            df_all = pd.concat([df, df_all])

    return df_all.reset_index()

def _get_best_results(df_all):

    for ii, idx in enumerate(set(df_all.idx)):
        df_sub = df_all[df_all.idx == idx]
        df_subB = df_sub[df_sub.norm == min(df_sub.norm)]

        if ii == 0:
            df_best = df_subB
        else:
            df_best = pd.concat([df_subB, df_best])

    return df_best.reset_index()

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
        ax[0].hist(df_sub.agr, bins=binsA, color='gray', alpha=10/num_cats)
        ax[1].hist(df_sub.bgr, bins=binsB, color='gray', alpha=10/num_cats)
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
        ctab = _make_ctab(compl_tables['perms'][int(cid)], years_chk, mags_chk)

        for ii in range(len(ctab)-1):
            plt.plot([ctab[ii][0], ctab[ii+1][0]], [ctab[ii][1], ctab[ii][1]], 'r', alpha=0.01)
            plt.plot([ctab[ii+1][0], ctab[ii+1][0]], [ctab[ii][1], ctab[ii+1][1]], 'r', alpha=0.01)
            plt.plot(ctab[ii+1][0], ctab[ii][1], 'ko', alpha=0.01)

            yrs.append(ctab[ii+1][0])
            mgs.append(ctab[ii][1])
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

def plt_a_b_density(df, figsdir, figname, weights=None):
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

def make_all_plots(resdir_base, compdir, figsdir_base, labels):
    for label in labels:
        resdir = os.path.join(resdir_base, label)
        figsdir = os.path.join(figsdir_base, label)
        df_all = _read_results(resdir)
        df_best = _get_best_results(df_all)
        _make_a_b_histos(df_all, df_best, figsdir)
        plt_compl_tables(compdir, figsdir, df_best)
        plt_a_b_density(df_best, figsdir, 'a-b-density_best.png')
        plt_a_b_density(df_all, figsdir, 'a-b-density_all.png')
        
        df_thresh = get_top_percent(df_thresh, 0.2)
        plt_a_b_density(df_all, figsdir, 'a-b-density_20percent.png')

