import re
import os
import sys
from plot_cross_section import plt_cs
import matplotlib.pyplot as plt

def main(argv):

    if len(argv) < 2:
        folder = './'
    else:
        folder = argv[1]

    fin = open(argv[0], 'r')
    for line in fin:

        aa = re.split('\s+', line.rstrip())
        olo = float(aa[0])
        ola = float(aa[1])
        lnght = float(aa[2])
        strike = float(aa[3])
        ids = aa[4]
        ini_fle = aa[5]

        fig = plt_cs(olo, ola, lnght, strike, ids, ini_fle)
        name = 'section_%s.pdf' % (ids)
        path = os.path.join(folder, name)
        fig.savefig(path, bbox_inches='tight')
        print ('Created %s' % (path))
        plt.close()

if __name__ == "__main__":
    main(sys.argv[1:])
