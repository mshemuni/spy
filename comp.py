from spy import Fits, FitsArray


def combine_compair(pref):
    print(pref)

    iraf = Fits.from_path(f"C:\\Users\\mshem\\Desktop\\20131108N\\compaire\\{pref}i.fits")
    spy = Fits.from_path(f"C:\\Users\\mshem\\Desktop\\20131108N\\compaire\\{pref}s.fits")
    if pref.startswith("flat"):
        diff = iraf / spy
    else:
        diff = iraf - spy

    print(diff.imstat().iloc[0])


def ccdproc():
    bias = FitsArray.from_pattern(r"C:\\Users\\mshem\\Desktop\\20131108N\\BDF\\Bias*.fits")
    dark = FitsArray.from_pattern(r"C:\\Users\\mshem\\Desktop\\20131108N\\BDF\\Dark_0.5000*.fits")
    flat = FitsArray.from_pattern(r"C:\\Users\\mshem\\Desktop\\20131108N\\BDF\\Flat*V.fits")

    master_zero = bias.zero_combine("median", "minmax")
    master_dark = dark.dark_combine("median", "minmax", weights=["EXPTIME"] * len(dark))
    master_flat = flat.flat_combine("median", "minmax", weights=["EXPTIME"] * len(flat))

    light = FitsArray.from_pattern(r"C:\\Users\\mshem\\Desktop\\20131108N\\CPCam\\CPCam*V.fits")

    corrected = light.zero_correction(master_zero).dark_correction(master_dark).flat_correction(master_flat)


if __name__ == '__main__':
    ccdproc()
    # lst = [
    #     "zero_a_m_",
    #     "zero_a_s_",
    #     "zero_m_m_",
    #     "zero_m_s_",
    #
    #     "dark_a_m_",
    #     "dark_a_s_",
    #     "dark_m_m_",
    #     "dark_m_s_",
    #
    #     "flat_a_m_",
    #     "flat_a_s_",
    #     "flat_m_m_",
    #     "flat_m_s_",
    # ]
    #
    # for each in lst:
    #     combine_compair(each)
