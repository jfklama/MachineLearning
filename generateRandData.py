import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return  1/(1 + np.exp(-(3*x-6)) )

def normal(x, mu, sigma):
    pdf = ( np.exp( (-( x - mu )**2) / (2 * sigma**2) ) / (sigma * np.sqrt(2 * np.pi)) )
    return pdf

def lognormal(x, mu, sigma):
    pdf = ( np.exp( (-( np.log(x) - mu )**2) / (2 * sigma**2) ) / (x * sigma * np.sqrt(2 * np.pi)) )
    return pdf

###############################################################################

def getRandomPt(nData):
    pT = np.random.uniform(20,100, nData)
    return pT

def getRandomEta(nData):
    Eta = np.random.uniform(0,2.3, nData)
    return Eta

###############################################################################
# Any functions work here; these are based on original FR shape

def getEtaPDF(Eta):
    #PDF = 0.05*sigmoid(Eta)
    PDF = normal(Eta, 1, 0.5)
    return PDF

def getPtPDF(pT):
    #PDF = 50*lognormal(pT, 0, 1)
    PDF = normal(pT, 50, 10)
    return PDF

###############################################################################

#create labels {0,1} based on given p.d.f.
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

#divide variable into intervals to make histo
def getBinnedVar(variable, nbins):
    binnedVar = np.empty(nbins)
    if(variable=='pT'):
        idx = 0
        while idx < nbins:
            binnedVar[idx] = idx*(80/nbins) + 20
            idx += 1
    if(variable=='Eta'):
        idx = 0
        while idx < nbins:
            binnedVar[idx] = idx*(2.3/nbins)
            idx += 1
    #else:
    #    print('Given variable does not exist!')

    return binnedVar

#count fake taus in created intervals (make histo)
def getFR_PtHisto(pT, labels, nbins, nData):

    pTsorted = getRandomPt(nData)
    pTsorted.sort()
    pT_bins = np.empty( nbins )
    trueFakesPt = np.zeros( int(len(pT)*nbins/(nData)) )

    idx = 0
    while idx < nbins:
        i = 0
        while i < len(pT):
            #if labels[i]==1:
                #print(idx, pT[i], labels[i])
            if( pT[i] >= pTsorted[ int(idx*(nData/nbins)) ] and pT[i] < pTsorted[ int((idx+1)*(nData/nbins))-1 ] ):
                if labels[i] ==1:
                    trueFakesPt[idx] += 1
            i += 1
            #print(idx)
        idx = idx + 1

    return trueFakesPt/(nData)

def getFR_EtaHisto(Eta, labels, nbins, nData):

    Etasorted = getRandomEta(nData)
    Etasorted.sort()
    Eta_bins = np.empty( nbins )
    trueFakesEta = np.zeros( int(len(Eta)*nbins/(nData)) )

    idx = 0
    i = 0
    while idx < nbins:
        i = 0
        while i < len(Eta):
            if(Eta[i] >= Etasorted[ int(idx*(nData/nbins)) ] and Eta[i] < Etasorted[ int((idx+1)*(nData/nbins))-1 ] ):
                if labels[i] ==1:
                    trueFakesEta[idx] += 1
            i += 1
        Eta_bins[idx] = idx*(2.3/nbins)
        idx += 1

    return trueFakesEta/(nData)

#################
#test plots
#################

'''
nData = 50000
nbins = 8

Eta = getRandomEta(nData)
pT = getRandomPt(nData)

PDF = getEtaPDF(Eta)*getPtPDF(pT)
#PDF = PDF / np.linalg.norm(PDF)
labels = getLabels(PDF)

plt.figure(1)
plt.scatter(pT, PDF, s=1)
plt.scatter( getBinnedVar('pT', nbins ), 10*getFR_PtHisto(pT, labels, nbins, nData) )
plt.legend(loc=0)
plt.show(block=False)

plt.figure(2)
plt.scatter(Eta, PDF, s=1)
plt.scatter( getBinnedVar('Eta', nbins ), 10*getFR_EtaHisto(Eta, labels, nbins, nData) )
plt.legend(loc=0)
plt.show()

#labels = PDF
#labels[ labels>=0.05 ]=1

print(PDF, getLabels(PDF))
plt.figure(1)
plt.scatter(pT, getPtPDF(pT), s=1)
plt.xlabel('pT')
plt.ylabel('prawdopodobienstwo')
#plt.scatter( pT, getLabels(PDF), s=2 )
plt.legend(loc=0)
plt.show(block=False)

plt.figure(2)
plt.scatter(Eta, getEtaPDF(Eta), s=1)
plt.xlabel('Eta')
plt.ylabel('prawdopodobienstwo')
#plt.scatter( Eta, getLabels(PDF), s=2 )
plt.legend(loc=0)
plt.show()
'''
