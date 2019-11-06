#Exercise 1

def convert(c):
    f = (c*1.8)+32
    if(f>75):
        print("Too hot!")
    else:
        degrees = (c,f)
        print(degrees)


convert(50)
convert(20)


#Exercise 2
def primeNumber(inputX):
    primeList = [x for x in range(2,100) if all(x%y!=0 for y in range(2,x))]
    print("Prime numbers are: ", primeList)

primeNumber(100)

#Exerise 3
import numpy as np
x = np.zeros((10,10))
x[1:-1,1:-1] = 1
print(x)