import gc
a=range(1000000*1000000)
del a
gc.collect()
print("over")