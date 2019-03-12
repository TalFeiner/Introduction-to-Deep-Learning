#!/usr/bin/env python

import numpy as np
#import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

plt.style.use('classic')
data = np.loadtxt("assign1_data.txt",skiprows = 2)

def part_1():
    global data
    cb_1 = 0; cr_1 = 0; Xr_1 = np.zeros(len(data[:,0])); Xb_1 = np.zeros(len(data[:,0])); Yr_1 = np.zeros(len(data[:,0])); Yb_1 = np.zeros(len(data[:,0]))
    cb_2 = 0; cr_2 = 0; Xr_2 = np.zeros(len(data[:,0])); Xb_2 = np.zeros(len(data[:,0])); Yr_2 = np.zeros(len(data[:,0])); Yb_2 = np.zeros(len(data[:,0]))
    
    for ii in range(100):
        S_xy1 = (data[ii,0]-data[:,0]) * (data[ii,2]-data[:,2])
        S_xx1 = (data[ii,0]-data[:,0])
        S_xy2 = (data[ii,1]-data[:,1]) * (data[ii,2]-data[:,2]) 
        S_xx2 = (data[ii,1]-data[:,1])
        
    m_1 = np.sum(S_xy1) / np.sum(S_xx1**2)
    m_2 = np.sum(S_xy2) / np.sum(S_xx2**2)
    b_1 = np.mean(data[:,2]) - m_1*np.mean(data[:,0])
    b_2 = np.mean(data[:,2]) - m_2*np.mean(data[:,1])
    
    Y_1 = m_1*data[:,0] + b_1
    Y_2 = m_2*data[:,1] + b_2
    
    for ii in range(100):
        if Y_1[ii] > data[ii,2]:
            Xb_1[cb_1] = data[ii,0]
            Yb_1[cb_1] = data[ii,2]
            cb_1 = 1+cb_1
        elif Y_1[ii] < data[ii,2]:
            Xr_1[cr_1] = data[ii,0]
            Yr_1[cr_1] = data[ii,2]
            cr_1 = 1+cr_1
        
        if Y_2[ii] > data[ii,2]:
            Xb_2[cb_2] = data[ii,1]
            Yb_2[cb_2] = data[ii,2]
            cb_2 = 1+cb_2
        elif Y_2[ii] < data[ii,2]:
            Xr_2[cr_2] = data[ii,1]
            Yr_2[cr_2] = data[ii,2]
            cr_2 = 1+cr_2
    
    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    axs[0].plot(Xb_1[:], Yb_1[:], 'b.', Xr_1[:], Yr_1[:], 'r.', data[:,0], Y_1, 'g')
    axs[0].set_title('Part 1')
    axs[0].set_xlabel('x_1')
    axs[0].set_ylabel('y')
    
    axs[1].plot(Xb_2[:], Yb_2[:], 'b.', Xr_2[:], Yr_2[:], 'r.', data[:,1], Y_2, 'g')
    axs[1].set_xlabel('x_2')
    axs[1].set_ylabel('y')
    
    plt.show()

def part_2():
    global data
    
    A = np.vstack([data[:,0], data[:,1], np.ones(len(data[:,0]))]).T
    w_1, w_2, b = np.linalg.lstsq(A, data[:,2], rcond=-1)[0]
    print ("part 2: w_1 =", w_1, "w_2 =", w_2,"b =", b)
    return w_1, w_2, b
    
def part_3(w_1, w_2, b):
    
    P = part(w_1, w_2, b, 100)    
    print ("part 3 percentage :", P,'%')
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    
    #for c, m, zlow, zhigh in [('r', 'o', -100, -100), ('b', '^', -0, -0)]:
     #   ax.scatter(data[:,0], data[:,1], data[:,2], c=c, marker=m)
    
   # plt.show()
    
def part(w_1, w_2, b, n):
    global data
    p = np.zeros(n); 
    
    for ii in range(n):
        
        C = w_1*data[(100-n)+ii,0] + w_2*data[(100-n)+ii,1] + b 
        if C > 0:
            z_1 = 1
        else:
            z_1 = 0

        if z_1 == data[(100-n)+ii,3]:
            p[ii] = 1
        else:
            p[ii] = 0
    
    P = (np.sum(p)/n) * 100
    return P

def part_4():
    global data
    
    #Train the model on the first {25} examples
    A = np.vstack([data[0:24,0], data[0:24,1], np.ones(len(data[0:24,0]))]).T
    w_1, w_2, b = np.linalg.lstsq(A, data[0:24,2], rcond=-1)[0]
    
    P_25 = part (w_1, w_2, b, 75)
    print ("success percentage for model which trained on the first 25 examples :", P_25,'%')
    
    #Train the model on the first {50} examples
    A = np.vstack([data[0:49,0], data[0:49,1], np.ones(len(data[0:49,0]))]).T
    w_1, w_2, b = np.linalg.lstsq(A, data[0:49,2], rcond=-1)[0]
    
    P_50 = part (w_1, w_2, b, 50)
    print ("ssuccess percentage for model which trained on the first 50 examples :", P_50,'%')

    #Train the model on the first {25} examples
    A = np.vstack([data[0:74,0], data[0:74,1], np.ones(len(data[0:74,0]))]).T
    w_1, w_2, b = np.linalg.lstsq(A, data[0:74,2], rcond=-1)[0]
    
    P_75 = part (w_1, w_2, b, 25)
    print ("success percentage for model which trained on the first 75 examples :", P_75,'%')

    #w1=w2=b=0.
    P_0 = part (0, 0, 0, 100)
    print ("success percentage when  w1=w2=b=0 :", P_0,'%')

if __name__ == "__main__":
    
    part_1 ()
    w_1, w_2, b  = part_2 ()
    part_3 (w_1, w_2, b)
    part_4 ()
    #print (w_1, w_2, b, A)
    
    pass
