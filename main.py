
from spy import Fits

fits = Fits.sample()
print(fits.crop(1000, 1000, 1, 1).data().size)
