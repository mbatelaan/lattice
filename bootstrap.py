import numpy as np

startseed = 1234


def bootstrap(raw_data, config_ax, nboot, nbin=1):
    """Take a matrix with the configurations along the config_ax axis and return a bootstrapped version of the matrix with the bootstraps along axis 0


    nboot = number of bootstrap resamples
    nbin = number of bins (default 1)
    """
    nconf = int(np.shape(raw_data)[config_ax] / nbin)
    # move the axis with the configurations to axis 0
    data = np.moveaxis(raw_data, config_ax, 0)
    bsdata = np.empty([nboot, *np.shape(data)[1:]])  # Results array
    if nbin > 1:  # sort the data into bins and average over the bin values
        tmpdata = np.array(
            [
                np.average(ien)
                for ien in np.split(
                    np.array(data[0 : nbin * (int(len(data) / nbin))]),
                    self.nconf,
                )
            ]
        )
    else:
        tmpdata = np.array(data)
    myseed = startseed * nconf / nboot
    np.random.seed(int(myseed))
    locranint = np.random.randint
    for iboot in range(nboot):
        rint = locranint(0, nconf, nconf)
        bsdata[iboot] = np.average(tmpdata[rint], axis=0)
    return bsdata


# def bootstrap(array, nboot, cfg_axis=0, boot_axis=0):
#     array = np.moveaxis(array, cfg_axis, 0)
#     length = np.shape(array)[0]
#     result = np.empty([nboot, *np.shape(array)[1:]], dtype=array.dtype)
#     result[0] = np.mean(array, axis=0)
#     myseed = int(startseed * nboot)
#     np.random.seed(myseed)

#     for iboot in range(1, nboot):
#         rint = np.random.randint(0, length, size=length)
#         result[iboot] = np.mean(array[rint], axis=0)
#     result = np.moveaxis(result, 0, boot_axis)
#     return result
