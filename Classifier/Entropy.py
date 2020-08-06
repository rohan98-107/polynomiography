from PIL import Image
import numpy as np

def selectThreshold(fp):
    img = Image.open(fp).convert('LA')
    imgarr = np.asarray(img)
    #img.save('greyscale.png')
    matrix = [[p[0] for p in r] for r in imgarr]

    T = matrix[0][0]
    R1 = set()
    R2 = set()
    mu1 = 0
    mu2 = 0

    x = 0
    for r in matrix:
        y = 0
        for p in r:
            if p > T:
                R1.add((x,y))
                mu1 += p
            else:
                R2.add((x,y))
                mu2 += p
            y += 1
        x += 1

    mu1 = mu1/len(R1)
    mu2 = mu2/len(R2)


    T_new = (mu1 + mu2)/2
    T_old = 0
    while T_new != T_old:
        i1 = 1
        i2 = 1
        sum1 = 0
        sum2 = 0
        for r in matrix:
            for p in r:
                if p > T_new:
                    i1 += 1
                    sum1 += p
                else:
                    i2 += 1
                    sum2 += p

        mu1 = sum1/i1
        mu2 = sum2/i2

        T_old = T_new
        T_new = (mu1 + mu2)/2

    return T_new

def detectEdges(fp,mask_length=3,mask_width=3):
    img = Image.open(fp).convert('LA')
    imgarr = np.asarray(img)
    matrix = [[p[0] for p in r] for r in imgarr]
    T = selectThreshold(fp)
    binimg = matrix

    x = 0
    for r in matrix:
        y = 0
        for p in r:
            if p > T:
                binimg[x][y] = 1
            else:
                binimg[x][y] = 0
            y += 1
        x += 1

    M = mask_length
    N = mask_width

    a = int((M-1)/2)
    b = int((N-1)/2)

    window = np.zeros((M,N,2))
    for i in range(-a,a+1):
        for j in range(-b,b+1):
            window[i+a,j+b] = [i,j]

    for y in range(b+1,N-b):
        for x in range(a+1,M-a):
            sum = 0
            # little confused here - this is the main algorithm 

#dir = "/Users/rohanrele/Documents/research/Polynomiography/Classifier/images/"
#detectEdges(dir + "cat_pic.png")
