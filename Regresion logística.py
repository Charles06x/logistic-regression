# -*- coding: utf-8 -*-
import numpy as np
import math
import matplotlib as mp
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data

#Load Dataset
dataSet=np.loadtxt("datasetRegLog.txt",delimiter=';')
x=dataSet[:,[0,1]]
y=dataSet[:,2]

thetas = [0,0,0]; alpha = 0.0009
for n in range(0,1000):
    aux = [0,0,0]; sum = 0

    for j in range(0,3):
        for i in range(0, int(len(x)*0.6)):

            e = (((thetas[0] + thetas[1] * x[i][0] + thetas[2] * x[i][1])))
            e = 1/(1 + math.exp(-e))
            if j == 0:
                sum = sum + ( ( ( e - y[i] ) ) ) #Xo = 1
            else:
                sum = sum +  ( e - y[i] ) * x[i][j-1]
        #print("Sumatoria: " + str(n) + "| " + str(j) + ": ",  sum )
        aux[j] = thetas[j] - (alpha/int(len(x)*0.8)) * sum
    thetas = aux
print(thetas)
print("\n##########################\n")
s = ("Formula: " + str(thetas[0]) + " + " + str(thetas[1]) + "*X + " + str(thetas[2]) + "*Y"   )
print(s)
####Test the resultant formula
dsq = 0
tp, tn, fp, fn = 0, 0, 0, 0     #Confusion matrix
for n in range(int(len(x)*0.6), len(x)):
    g = thetas[0] + thetas[1] * x[n][0]  + thetas[2] * x[n][1]
    g = 1/(1 + math.exp(-g))
    print(g, "| " +  str(y[n]))
    if g < 0.5:     #If predicted is negative...
        if y[n] == 0:   #And actual is negative
            tn += 1
        if y[n] == 1:   #and actual is positive
            fn += 1
    else:           #If predicted is positive...
        if y[n] == 1:   #And acutal is positive
            tp += 1
        if y[n] == 0:   #And Actual is negative
            fp += 1
    dsq += (g - y[n])**2

error = (dsq/(len(x)-int(len(x)*0.6)))
print("Error: ", error)

#####Precision, Recall and F1-Score
##Precision:
precision = tp/(tp + fp)
##Recall:
recall = tp/(tp + fn)
##F1-Score
f_score = (2 * precision * recall)/(precision + recall)

print("#########################################")
print("#########################################")
print("\t   CONFUSION MATRIX")
print("         Negative     Positive")
print("Negative   {0}           {1}".format(tn,fp))
print("Positive   {0}           {1}".format(fn,tp))
print("#########################################")
print("#########################################")

print("Precision: ",precision)
print("Recall: ",recall)
print("f_score: ", f_score)
###Charles Acevedo
