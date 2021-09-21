import numpy as np
import multiprocessing
from itertools import product


def mm(a):
    return a


def get_completenesses():

    tmp = [1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 1995, 2000, 2005, 2010]
    # tmp = [1900, 1930, 1960, 1970, 1980]
    # tmp = [1900, 1930, 1960, 1970]
    years = np.array(tmp)
    mags = np.array([4.0, 4.5, 5, 5.5, 6, 6.5, 7])
    #mags = np.array([4.0, 5.0, 6.0, 7.0])
    idxs = np.arange(len(mags))
    idxs[::-1].sort()
    max_first_idx = 5

    print('Total number of combinations: {:,d}'.format(len(mags)**len(years)))

    step = 4
    perms = []
    for y in [years[i:min(i+step, len(years))] for i in range(0, len(years), step)]:
        with multiprocessing.Pool(processes=8) as pool:
            p = pool.map(mm, product(idxs, repeat=len(y)))
            p = np.array(p)
            p = p[np.diff(p, axis=1).min(axis=1) >= 0, :]
            if len(perms):
                new = []
                for x in perms:
                    for y in p:
                        new.append(list(x)+list(y))
                perms = new
            else:
                perms = p
        p = np.array(perms)
        p = p[np.diff(p, axis=1).min(axis=1) >= -1e-10, :]
        p = p[p[:, 0] <= max_first_idx]
        perms = p

    np.save('dispositions.npy', perms)
    np.save('mags.npy', mags)
    np.save('years.npy', years)
    print(len(perms))


if __name__ == '__main__':
    get_completenesses()


