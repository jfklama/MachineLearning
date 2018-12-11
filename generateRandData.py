import numpy as np
import matplotlib.pyplot as plt

def inv_sigmoid(x):
    #v = exp(-x)
    return  np.exp(-x)

def sigmoid(x):
    #v = exp(-x)
    return  1/(1 + np.exp(-(3*x-6)) )

def lognormal(x, mu, sigma):
    pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi)))
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
    PDF = 500*lognormal(pT, 1, 1)
    return PDF / np.linalg.norm(PDF)
    #return PDF

def getLabels(PDF):
    labels = PDF
    for i in labels:
        if i >= 0.5:
            i = 1.
        if i<0.5:
            i = 0.
    return labels

'''
nData = 50000

Eta = getRandomEta(nData)
#y1 = getEtaPDF(Eta)

pT = getRandomPt(nData)
y2 = getPtPDF(pT)

PDF = getEtaPDF(Eta)+getPtPDF(pT)
PDF = PDF / np.linalg.norm(PDF)

labels = PDF
labels[ labels>=0.05 ]=1

print(PDF, labels)
plt.plot( pT, PDF )
plt.show()
'''
