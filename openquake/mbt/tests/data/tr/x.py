import numpy
fou = open('crust.xyz', 'w')
for lo in numpy.arange(9, 11, 0.1):
    for la in numpy.arange(44, 46, 0.1):
        fou.write('{:.2f} {:.2f} {:.2f}\n'.format(lo, la, 30))
fou.close()
