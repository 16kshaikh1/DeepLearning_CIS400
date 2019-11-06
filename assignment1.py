import numpy as np
import time

# - Fill in the code below the comment Python and NumPy same as in example.
# - Follow instructions in document.
###################################################################
# Example: Create a zeros vector of size 10 and store variable tmp.
# Python
pythonStartTime = time.time()
tmp_1 = [0 for i in range(10)]
pythonEndTime = time.time()

# NumPy
numPyStartTime = time.time()
tmp_2 = np.zeros(10)
numPyEndTime = time.time()
#print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
#print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))


z_1 = None
z_2 = None
################################################################
# 1. Create a zeros array of size (3,5) and store in variable z.
# Python
pythonStartTime = time.time()


z_1 = [0] * 3
for i in range(3):
    z_1[i] = [0] * 5

pythonEndTime = time.time()
#print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
#print(z_1)

# NumPy
numPyStartTime = time.time()
z_2 = np.zeros((3,5))
numPyEndTime = time.time()
#print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))

#print(z)


#################################################
# 2. Set all the elements in first row of z to 7.
# Python
pythonStartTime = time.time()
for i in range(5):
    z_1[0][i] = 7
pythonEndTime = time.time()
#print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))



#print(z)

# NumPy
numPyStartTime = time.time()
z_2[0] = 7
numPyEndTime = time.time()
#print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))


#print(z)


#####################################################
# 3. Set all the elements in second column of z to 9.
# Python
pythonStartTime = time.time()
for i in range(3):
    z_1[i][1] = 9
pythonEndTime = time.time()
#print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))

#print(z)


# NumPy
numPyStartTime = time.time()
z_2[:,1] = 9
numPyEndTime = time.time()
#print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))




#print(z)


#############################################################
# 4. Set the element at (second row, third column) of z to 5.
# Python
pythonStartTime = time.time()
z_1[1][2] = 5
pythonEndTime = time.time()
#print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
#print(z_1)


#print(z)

# NumPy
numPyStartTime = time.time()
z_2[1,2] = 5
numPyEndTime = time.time()
#print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))



#print(z)


##############
print(z_1)
print(z_2)
##############


x_1 = None
x_2 = None
##########################################################################################
# 5. Create a vector of size 50 with values ranging from 50 to 99 and store in variable x.
# Python
pythonStartTime = time.time()
x_1 = []
for i in range(50,100):
    x_1.append(i)
pythonEndTime = time.time()
#print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))

#print(x)


# NumPy
numPyStartTime = time.time()
x_2 = np.arange(50,100)
numPyEndTime = time.time()




#print(x)
##############
print(x_1)
print(x_2)
##############


y_1 = None
y_2 = None
##################################################################################
# 6. Create a 4x4 matrix with values ranging from 0 to 15 and store in variable y.
# Python

pythonStartTime = time.time()
y_1 = [[0]*4 for i in range(4)]
b = 0
for i in range(4):
    for j in range(4):
        y_1[i][j] = b
        b = b+1
pythonEndTime = time.time()
#print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))

#print(y_1)

# NumPy
numPyStartTime = time.time()
y_2 = np.arange(1,17).reshape(4,4)
numPyEndTime = time.time()



#print(y)


##############
print(y_1)
print(y_2)
##############


tmp_1 = None
tmp_2 = None
####################################################################################
# 7. Create a 5x5 array with 1 on the border and 0 inside amd store in variable tmp.
# Python
pythonStartTime = time.time()
n=5
tmp_1 = [0] * n
for i in range(n):
    tmp_1[i] = [0] * n
for i in range(5):
    tmp_1[0][i] = 1
    tmp_1[-1][i] = 1
    tmp_1[i][0] = 1
    tmp_1[i][-1] = 1
pythonEndTime = time.time()
#print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))


#print(tmp)




# NumPy
numPyStartTime = time.time()
tmp_2 = np.ones((5,5))
tmp_2[1:-1,1:-1] = 0
numPyEndTime = time.time()



#print(tmp)

##############
print(tmp_1)
print(tmp_2)
##############


a_1 = None; a_2 = None
b_1 = None; b_2 = None
c_1 = None; c_2 = None
#############################################################################################
# 8. Generate a 50x100 array of integer between 0 and 5,000 and store in variable a.
# Python
pythonStartTime = time.time()
a_1 = [[0]*100 for i in range(50)]
b = 0
for i in range(50):
    for j in range(100):
        a_1[i][j] = b
        b = b+1
pythonEndTime = time.time()


#print(a_1)

# NumPy
numPyStartTime = time.time()
a_2 = np.arange(0,5000).reshape(50,100)
numPyEndTime = time.time()




#print(a)

###############################################################################################
# 9. Generate a 100x200 array of integer between 0 and 20,000 and store in variable b.
# Python
pythonStartTime = time.time()
b_1 = [[0]*200 for i in range(100)]
c = 0
for i in range(100):
    for j in range(200):
        b_1[i][j] = c
        c += 1
pythonEndTime = time.time()
#print(b_1)

# NumPy
numPyStartTime = time.time()
b_2 = np.arange(0,20000).reshape(100,200)
numPyEndTime = time.time()




#print(b_2)

#####################################################################################
# 10. Multiply matrix a and b together (real matrix product) and store to variable c.
# Python
pythonStartTime = time.time()
c_1 = [[0] * 200 for i in range(50)]
for i in range(50):
    for j in range(100):
        for k in range(200):
            c_1[i][k] = c_1[i][k] + a_1[i][j] * b_1[j][k]
pythonEndTime = time.time()

#print(c_1)

# NumPy
#print("C_2")
numPyStartTime = time.time()
c_2 = np.matmul(a_2,b_2)
numPyEndTime = time.time()



#print(c_2)

d_1 = None; d_2 = None
################################################################################
# 11. Normalize a 3x3 random matrix ((x-min)/(max-min)) and store to variable d.
# Python
import random
pythonStartTime = time.time()
d_1 = [0] * 3
for i in range(3):
    d_1[i] = [0] * 3

for i in range(3):
    for j in range(3):
        d_1[i][j] = random.randint(1,100)

d_max = max(max(d_1))
d_min = min(min(d_1))

for i in range(3):
    for j in range(3):
        d_1[i][j] = (d_1[i][j]-d_min)/(d_max-d_min)


pythonEndTime = time.time()

# NumPy
numPyStartTime = time.time()

d_2 = np.random.randint(50,size=(3,3))
d_max,d_min = d_2.max(),d_2.min()
d_2 = (d_2-d_min)/(d_max-d_min)
numPyEndTime = time.time()




##########
print(d_1)
print(d_2)
#########


################################################
# 12. Subtract the mean of each row of matrix a.
# Python
pythonStartTime = time.time()
for i in range(len(a_1)):
    m_1 =sum(a_1[i])/len(a_1[i])
    for j in range(len(a_1[i])):
        a_1[i][j]= int(a_1[i][j]-m_1)
print(a_1)
pythonEndTime = time.time()



# NumPy
numPyStartTime = time.time()
mean_2 = np.mean(a_2,axis=1)
for i in range(len(mean_2)):
    a_2[i] = a_2[i] - mean_2[i]
print(a_2)
numPyEndTime = time.time()





###################################################
# 13. Subtract the mean of each column of matrix b.
# Python
pythonStartTime = time.time()
for i in range(len(b_1)):
    mean_1 =sum(b_1[i])/len(b_1[i])
    for j in range(len(b_1[i])):
        b_1[i][j]= int(b_1[i][j]-mean_1)
print(b_1)
pythonEndTime = time.time()



# NumPy
numPyStartTime = time.time()

mean_2 = np.mean(b_2,axis=1)
for i in range(len(mean_2)):
    b_2[i] = b_2[i]-mean_2[i]
print(b_2)
numPyEndTime = time.time()






################
print(np.sum(a_1 == a_2))
print(np.sum(b_1 == b_2))
################

e_1 = None; e_2 = None
###################################################################################
# 14. Transpose matrix c, add 5 to all elements in matrix, and store to variable e.
# Python
pythonStartTime = time.time()
e_1 = [0] * len(c_1[0])
for i in range(len(e_1)):
    e_1[i] = [0] * len(c_1)
for i in range(len(e_1)):
    for j in range(len(e_1[i])):
        e_1[i][j] = c_1[j][i]
        e_1[i][j] += 5
pythonEndTime = time.time()

#print(e_1)


# NumPy
numPyStartTime = time.time()
e_2 = np.transpose(c_2)
e_2 += 5
numPyEndTime = time.time()


#print(e_2)

##################
print (np.sum(e_1 == e_2))
##################


#####################################################################################
# 15. Reshape matrix e to 1d array, store to variable f, and print shape of f matrix.
# Python
pythonStartTime = time.time()

f = [j for sub in e_1 for j in sub]
pythonEndTime = time.time()


# NumPy
numPyStartTime = time.time()
f = e_2.flatten()
numPyEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))
#print(f.shape)


