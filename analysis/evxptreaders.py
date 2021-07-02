import numpy as np
from .bootstrap import bootstrap


def evxptdata(evxptres, numbers=[0], nboot=500, nbin=1):
    """Get the evxpt data from the file and output a bootstrapped numpy array.

    'numbers' is a list of the number of the result you want if there are multiple in the file
    The output is a numpy matrix with:
    axis=0: bootstraps
    axis=1: 'number' of data in parameter file
    axis=2: time axis
    axis=3: real & imaginary parts
    """
    # print(evxptres)
    corrlist = []
    for number in numbers:
        f = open(evxptres)
        for line in f:
            strpln = line.rstrip()
            if len(strpln) > 0:
                if strpln[0] == "+" and strpln[1] == "E" and strpln[2] == "N":
                    tmp = f.readline()
                    tmp = f.readline()
                    tmp = f.readline().split()
                    times = int(tmp[5])
                if (
                    strpln[0] == "+"
                    and strpln[1] == "R"
                    and strpln[2] == "P"
                    and int(strpln[4:6]) == number
                ):
                    tmp = f.readline().split()
                    while tmp[0] != "nmeas":
                        tmp = f.readline().split()
                    confs = int(tmp[2])
                    G = np.zeros(shape=(confs, times, 2))
                if (
                    strpln[0] == "+"
                    and strpln[1] == "R"
                    and strpln[2] == "D"
                    and int(strpln[4:6]) == number
                ):
                    for iff in range(confs):
                        for nt in range(times):
                            tmp = f.readline().split()
                            G[iff, nt, 0] = tmp[1]
                            G[iff, nt, 1] = tmp[2]
        f.close()
        corrlist.append(G)
        print(f"{confs=}")
        # print(np.shape(corrlist))
    corrlist = np.array(corrlist)
    # Pass corrlist to the bootstrap function
    bsdata = bootstrap(corrlist, config_ax=1, nboot=nboot, nbin=nbin)
    return bsdata
