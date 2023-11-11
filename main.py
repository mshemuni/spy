from spy import FitsArray

fa = FitsArray.sample()

print("=" * 20)
fa2 = fa.add(2)
fa2.save_as("data")
