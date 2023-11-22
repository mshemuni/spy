from spy import FitsArray, Fits

fa = FitsArray.sample()
fa2 = FitsArray.sample().shift(20, 102)


fa.merge(fa2)
fa.show()
