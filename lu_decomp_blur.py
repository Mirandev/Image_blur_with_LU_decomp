#Miranda Harrison

import numpy
import cv2
import scipy
import scipy.linalg
import math

def blur(img,blur_type):
    size = image.shape[0] # must be a square image
    output = numpy.zeros([size,size,3], numpy.uint8) #array to store the blurred image

    if blur_type == 'horizontal_motion':

        for i in range(size): #rows
            for j in range(size): #columns
                if (j-1)<0: #outside left border
                    jm1 = j
                else : jm1 = j-1
                if (j+1)>(size-1): #outside right border
                    jp1 = j
                else: jp1 = j+1
                output[i][j] = (img[i][jm1]/3 + img[i][j]/3 + img[i][jp1]/3).astype(numpy.uint8)

    else: # out of focus blur
        for i in range(size): # rows (n)
            for j in range(size): #columns
                if (j-1)<0: #outside left border
                    jm1 = j
                else : jm1 = j-1
                if (j+1)>(size-1): #outside right border
                    jp1 = j
                else: jp1 = j+1
                if (i-1)<0: #outside top border
                    above = i
                else: above = i - 1 
                if (i+1)>(size-1): #outside bottom border
                    below = i
                else: below = i + 1
                output[i][j] = (img[above][j]/8 + img[i][jm1]/8 + img[i][j]/2 + img[i][jp1]/8 + \
                    img[below][j]/8).astype(numpy.uint8)


    return output

def multiply(a,b): #multiplication for square matrices
    return [[sum(i * j for i, j in zip(b_r, a_c)) for a_c in zip(*a)] for b_r in b]

def lu_matrix(img):
    PA = img
    size = len(img)
    L = numpy.zeros([size,size])
    U = numpy.zeros([size,size])

    #creates L and U matricies 
    for i in range(size):
        L[i][i] = 1

        max_val = abs(U[i][i])
        max_row = i
        for k in range(i+1,size):
            if (abs(PA[k][i])>max_val):
                max_val = abs(PA[k][i])
                max_row = k
        PA[[max_row,i]] = PA[[i,max_row]] #swap the max row with row i
        

        for j in range(i+1):
            sum1 = 0
            for n in range(j):
                sum1 += U[n][i] * L[j][n]
            U[j][i] = PA[j][i] - sum1 

        for j in range(i,size):
            sum2 = 0
            for n in range(i):
                sum2 += U[n][i] * L[j][n]
            if(U[i][i]==0): # prevent divde by 0 errors
                L[j][i] = 0
            else: L[j][i] = (PA[j][i] - sum2) / U[i][i]
        L[i][i] = 1

    return (L,U)


image = cv2.imread('gray_smallest.jpg') #put image file name here

size = image.shape[0]

#comment out one of the below two lines
blur_type = 'horizontal_motion' # for horizontal motion blur
#blur_type = 'out_of_focus' #for out of focus blur

print("creating blurred image...")

blurred = blur(image,blur_type) 

cv2.imwrite('blurred_'+blur_type+'.jpg',blurred)

b = numpy.mean(blurred, axis = 2).flatten() # 1D vector of all the greyscale values of the columns of the blurred image

#creating the blur matrix
blur_matrix = numpy.zeros([size*size,size*size])
if blur_type == 'horizontal_motion':
    for i in range(size*size): #  creates horizontal motion blur matrix
        for j in range(size*size):
            if (i-1)<0: #outside left border
                im1 = i
            else: im1 = i-1
            if (i+1)>(size+1): #outside right border
                ip1 = i
            else: ip1 = i+1

            if i==j: # place the ones
                blur_matrix[i][im1] = 1/3
                blur_matrix[i][i] = 1/3
                blur_matrix[i][ip1] = 1/3
else: # out of focus blur matrix
    for i in range(size*size):
        for j in range(size*size):
            if (i-1)<0: #outside left border
                im1 = i
            else: im1 = i-1
            if (i+1)>(size+1): #outside right border
                ip1 = i
            else: ip1 = i+1
            if (i-1)<0: #outside top border
                above = i
            else: above = i - 1 
            if (i+1)>(size-1): #outside bottom border
                below = i
            else: below = i + 1

            if i==j: # place the values to create the blur matrix
                blur_matrix[i][im1] += 1/8
                blur_matrix[i][i] += 1/2
                blur_matrix[i][ip1] += 1/8  
                blur_matrix[above][j] +=1/8
                blur_matrix[below][j] +=1/8


print("created blurred image and matrix using " + blur_type)

#print(blur_matrix)

print("starting LU decomposition...")
L, U = lu_matrix(blur_matrix)
print("finished LU decomposition")

size2 = size*size #length of the vector b, d, and x
x = numpy.zeros([size2])
d = numpy.zeros([size2])

print('starting solving the equations...')

#Ld=b using forward subsitution
for i in range(size2):
    sum1 = 0
    for k in range(i):
        sum1 += L[i][k] * d[k]
    d[i] = b[i] - sum1

#Ux=d using backwards subsitution
for i in range(size2-1,-1,-1):
    sum2 = 0
    for k in range(i+1,size2):
        sum2 += U[i][k] * x[k]
    if(U[i][i]==0):
        x[i] = 0
    else: x[i] = (1/U[i][i]) * (d[i] - sum2)
print('finished solving the equations') 

#print(x)

# format the output in a way it can be saved/displayed
x_reshape = x.reshape(size,size)
output = numpy.zeros([size,size,3])
for i in range(size): 
    for j in range(size):
        output[i][j][0] = x_reshape[i][j].astype(numpy.uint8)
        output[i][j][1] = x_reshape[i][j].astype(numpy.uint8)
        output[i][j][2] = x_reshape[i][j].astype(numpy.uint8)
#print(output)
output = output.astype(numpy.uint8)
cv2.imwrite('output.jpg',output)

print('displaying images')
cv2.imshow('original image',image)
cv2.imshow('blurred image '+ blur_type,blurred)
cv2.imshow('output',output)
cv2.waitKey()
