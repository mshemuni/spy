import math

from spy import Fits, FitsArray

fa = FitsArray.sample()


angles = list(each * math.pi / 180 for each in range(0, 360, 36))
fa_r = fa.rotate(angles)
fa_r.show()
