from spy import FitsArray, Fits

fa = FitsArray.sample()
print(fa.hselect(["NAXIS"]))
