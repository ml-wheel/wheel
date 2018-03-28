from numpy import *

for i in range(1000):
    x = int(random.uniform(0, 100))
    y = int(random.uniform(0, 100))
    if x > 60 and y > 60:
        z = 1
    else:
        z = 0
    print(str(x)+","+str(y)+","+str(z))