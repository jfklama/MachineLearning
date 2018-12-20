import numpy as np
import matplotlib.pyplot as plt

def inv_sigmoid(x):
    #v = exp(-x)
    return  np.exp(-x)

def sigmoid(x):
    #v = exp(-x)
    return  1/(1 + np.exp(-(3*x-6)) )

def lognormal(x, mu, sigma):
    pdf = (np.exp(-(np.log(x - mu))**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi)))
    return pdf


def getRandomPt(nData):
    pT = np.random.uniform(20,100, nData)
    pT.sort()
    return pT

def getRandomEta(nData):
    Eta = np.random.uniform(0,2.3, nData)
    Eta.sort()
    return Eta

def getEtaPDF(Eta):
    #Eta = np.random.uniform(0,2.2, nData)
    PDF = sigmoid(Eta)
    return PDF / np.linalg.norm(PDF)
    #return PDF

def getPtPDF(pT):
    #Eta = np.random.uniform(0,2.2, nData)
    PDF = 500*lognormal(pT, 0, 1)
    return PDF / np.linalg.norm(PDF)
    #return PDF

def getLabels(PDF):
    labels = np.copy(PDF)
    idx = 0
    while idx < len(labels):
        randNum = np.random.uniform(0,1)
        if randNum <= labels[idx]:
            labels[idx] = 1
        else:
            labels[idx] = 0
        idx += 1
    return labels

'''
nData = 50000

Eta = getRandomEta(nData)
#y1 = getEtaPDF(Eta)

pT = getRandomPt(nData)
y2 = getPtPDF(pT)

PDF = getEtaPDF(Eta)+getPtPDF(pT)
PDF = PDF / np.linalg.norm(PDF)

#labels = PDF
#labels[ labels>=0.05 ]=1

print(PDF, getLabels(PDF))
plt.figure(1)
plt.scatter(pT, PDF, s=1)
plt.scatter( pT, getLabels(PDF), s=2 )
plt.legend(loc=0)
plt.show(block=False)

plt.figure(2)
plt.scatter(Eta, PDF, s=1)
plt.scatter( Eta, getLabels(PDF), s=2 )
plt.legend(loc=0)
plt.show()
'''
