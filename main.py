from spy import Fits
from astropy.coordinates import SkyCoord
from astropy import units

sky = SkyCoord(ra=85.32261894*units.degree, dec=-2.52717821*units.degree)

f = Fits.sample()
# print(f.pixels_to_skys(275, 200).iloc[0].sky)
b = f.bin(5)

xys = f.skys_to_pixels(sky)
xysb = b.skys_to_pixels(sky)

f.show(sources={"xcentroid": xys.iloc[0].xcentroid, "ycentroid": xys.iloc[0].ycentroid})
b.show(sources={"xcentroid": xysb.iloc[0].xcentroid, "ycentroid": xysb.iloc[0].ycentroid})



