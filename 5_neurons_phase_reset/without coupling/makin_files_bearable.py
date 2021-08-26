import os, csv

if not os.path.exists('without_coupling_short.dat'):
    os.mknod('without_coupling_short.dat')

f_out = open('without_coupling_short.dat', 'w')

with open('without_coupling.dat', 'r') as f_in:
    for line in f_in :
        time = float(line.rstrip().split(' ')[0])
        print(time)
        if (time >= 19 and time <= 53) :
            f_out.write(line)

f_in.close()
f_out.close()
