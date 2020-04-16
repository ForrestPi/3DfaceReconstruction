import os
with open('out.obj', 'r') as f:
    for line in f:
        if 'v' in line:
            print(line.strip())
        if 'f' in line:
            tmp = line.strip().split()
            print('%s %d %d %d' % (tmp[0], int(tmp[1]) - 1, int(tmp[2]) - 1, int(tmp[3]) - 1))
