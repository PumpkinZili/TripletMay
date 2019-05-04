import os
import sys
c = open('/data0/zili/code/triplet/data/cifar100/train.csv','w+')
c.write('id'+','+'name'+','+'class'+'\n')
for p, d, f in os.walk('/data0/zili/code/triplet/data/cifar100/train2'):
    i = 0
    for dir in d:
        for r, dd, filename in os.walk(os.path.join(p, dir)):
            for file in filename:
                c.write(file[:-4]+','+ dir+','+ str(i)+'\n')
        i += 1
c.close()
sys.exit(1)