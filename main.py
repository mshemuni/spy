from spy import FitsArray, Fits

# fits = FitsArray(
#     [Fits.from_data_header(each.data()) for each in FitsArray.sample()]
# )

fits = FitsArray.sample()
sources = fits.extract()

phot = fits.photometry(
    sources["xcentroid"].tolist(), sources["ycentroid"].tolist(),
    10
)
print(phot.sky)
