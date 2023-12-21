from astropy import units
from astropy.coordinates import SkyCoord

from spy import FitsArray

fa = FitsArray.sample()
sc = SkyCoord(ra=85.39916173 * units.degree, dec=-2.58265558 * units.degree)

pixel = fa.skys_to_pixels(sc)
# print(pixel[["xcentroid", "ycentroid"]].to_numpy().tolist())
print(pixel[["xcentroid", "ycentroid"]].to_numpy().tolist())