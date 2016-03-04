'''
Created on Nov 4, 2010
Chapter 5 source file for Machine Learing in Action
@author: Peter
'''
from __future__ import division
from numpy import *
from time import sleep
from os import listdir

def fileSize(filename):
    fr=open(filename)
    rowNums=int(len(fr.readlines()))
    fr=open(filename)
    colNums=int(len(fr.readline()))-1
    return rowNums,colNums

def image2vec(filename):
    fr=open(filename)
    returnvec=[]
    for row in fr.readlines():
            returnvec.extend(row.strip())    
    return returnvec

def chooseIndex1(u,Y,index2):
    absE=list(abs(u-Y-u[index2]+Y[index2]))
    index1=absE.index(max(absE))
    return index1
    
def inner_choose(X1,X2,kern):
    gama=1/2
    r=0
    d=3
    if kern == 'lin':
        K=inner(X1,X2)
    elif kern == 'rbf':
        K=exp(-gama*sum((X1-X2)*(X1-X2).T))
    elif kern == 'pol':
        K=(gama*inner(X1,X2)+r)**d
    elif kern == 'sig':
        K=tanh(gama*inner(X1,X2)+r)
    else: 
        print kern        
        print 'Kernel choose wrong!\n'
    return K
    
    
def update1(alpha,index1,index2,X,Y,C,w,b,kern):
    alpha1_old=copy(alpha[index1])
    alpha2_old=copy(alpha[index2])    
    
    L=max(0.,alpha[index2,0]+alpha[index1,0]-C)
    H=min(C,alpha[index2,0]+alpha[index1,0])
    if (Y[index1,0] != Y[index2,0]):
        L=max(0.,alpha[index2]-alpha[index1])
        H=min(C,C+alpha[index2]-alpha[index1])

    eit=inner_choose(X[index1],X[index1],kern)+inner_choose(X[index2],X[index2],kern)-2*inner_choose(X[index1],X[index2],kern)
    E1=X[index1]*w.T-b-Y[index1]
    E2=X[index2]*w.T-b-Y[index2]
    if eit<=0 : return w,b,alpha 
    deltE=E1-E2    
    temp_alpha2=alpha[index2]+Y[index2]*deltE/eit
    
    if temp_alpha2<=L:
        temp_alpha2=L
    elif temp_alpha2>=H: 
        temp_alpha2=H       
    alpha[index2]=temp_alpha2
    alpha[index1]=alpha1_old+Y[index1]*Y[index2]*(alpha2_old-alpha[index2])
    
    b1=b+(E1+Y[index1]*(alpha[index1]-alpha1_old)*inner_choose(X[index1],X[index1],kern)
        +Y[index2]*(alpha[index2]-alpha2_old)*inner_choose(X[index1],X[index2],kern))
    b2=b+(E2+Y[index1]*(alpha[index1]-alpha1_old)*inner_choose(X[index1],X[index1],kern)
        +Y[index2]*(alpha[index2]-alpha2_old)*inner_choose(X[index2],X[index2],kern))
    b=(b1+b2)/2
    if alpha[index1]>0 and alpha[index1]<C:
        b=b1
    elif alpha[index2]>0 and alpha[index2]<C:
        b=b2       
    w_old=w
    w=w_old+Y[index1]*(alpha[index1]-alpha1_old)*X[index1]+Y[index2]*(alpha[index2]-alpha2_old)*X[index2]
    return w,b,alpha 
    
def notEquivalentM(alpha1,alpha2,indices):
    A=alpha1.A
    B=alpha2.A
    for i in indices:
        if A[i,0]!=B[i,0]:
            return 1
    return 0 

def smo(X,Y,rows,cols,kern):
    C=200
    b=0
    w=mat(zeros(cols))
    alpha=mat(zeros(rows)).T    
    X=mat(X)
    Y=mat(Y).T
    
    u=X*w.T-b
    for indexr in range(rows):
        if not((alpha[indexr,0]==0 and Y[indexr,0]*u[indexr,0]>=1)):              
            u=X*w.T-b
            index2=indexr
            index1=chooseIndex1(u,Y,index2)
            w,b,alpha=update1(alpha,index1,index2,X,Y,C,w,b,kern)
    
    times2=0
    iter2=1
    while iter2:
        alpha_old=alpha
        iter1=1
        times1=0
        while iter1:
            indexM=[]
            alpha_old1=alpha
            u=X*w.T-b
            for indexr in range(rows):            
                if (alpha[indexr,0]>0 and alpha[indexr,0]<C and Y[indexr,0]*u[indexr,0]==1):
                    indexM.append(indexr)
                    index2=indexr
                    index1=chooseIndex1(u,Y,index2)
                    w,b,alpha=update1(alpha,index1,index2,X,Y,C,w,b,kern)
                    u=X*w.T-b
            #print indexM
            iter1=notEquivalentM(alpha_old1,alpha,indexM)
            times1+=1
        
        for indexr in range(rows):
            index2=indexr
            index1=chooseIndex1(u,Y,index2)
            w,b,alpha=update1(alpha,index1,index2,X,Y,C,w,b,kern)
            u=X*w.T-b
        
        iter2=notEquivalentM(alpha_old,alpha,range(rows))     
        
        times2+=1
        if times2>20:break                
    return b,w
    
def main():
    trainingList=listdir('trainingDigits')
    trainSize=len(trainingList)
    rows,cols=fileSize('trainingDigits/%s' % trainingList[0])
    cols=rows*cols
    trainingData=zeros((trainSize,cols))
    trainingLabels=ones(trainSize)
    for index in range(trainSize):
        filename=trainingList[index]    
        trainingData[index]=image2vec('trainingDigits/%s' % filename)
        if (filename.split('_')[0]!='1'):
            trainingLabels[index]=-1
    
    testingList=listdir('testDigits')
    testSize=len(testingList)
    rows,cols=fileSize('testDigits/%s' % testingList[0])
    cols=rows*cols
    testingData=zeros((testSize,cols))
    testingLabels=ones(testSize)
    for index in range(testSize):
        filename=testingList[index]    
        testingData[index]=image2vec('testDigits/%s' % filename)
        if (filename.split('_')[0]!='1'):
            testingLabels[index]=-1
    A=mat(testingData)
            
    for kern in ('lin','rbf','pol','sig'):      
        b,w=smo(trainingData,trainingLabels,trainSize,cols,kern)
        testlabels_rec=zeros(testSize)-1
        pn=A*w.T-b
        count=0
        for index in range(testSize):
            if pn[index,0]>0:
                testlabels_rec[index]=1    
            if not testingLabels[index]==testlabels_rec[index]:
                count+=1
        accu=(1-count/testSize)*100    
        print "The accuracy  of the '%s' kernel classifier is:%s%%" % (kern,accu)
        
if __name__=='__main__':
    main()
      