from spy import Fits

sample = Fits.sample()

sample.hedit("TEST", 22)
print(sample.header())
