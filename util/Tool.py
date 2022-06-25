from util.config import LineConfig
from collections import defaultdict
import random

class Rating(object):
    'data access control'
    def __init__(self,config,trainingSet, testSet):
        self.config = config
        self.evalSettings = LineConfig(self.config['evaluation.setup'])
        self.miRNA = {} #map miRNA names to id
        self.drug = {} #map drug names to id
        self.id2miRNA = {}
        self.id2drug = {}
        self.miRNAMeans = {} #mean values of miRNAs's ratings
        self.drugMeans = {} #mean values of drugs's ratings
        self.globalMean = 0
        self.trainSet_u = defaultdict(dict)
        self.trainSet_i = defaultdict(dict)
        self.testSet_u = defaultdict(dict)
        self.testSet_i = defaultdict(dict)
        self.rScale = []
        self.trainingData = trainingSet[:]
        self.testData = testSet[:]
        self.__generateSet()
        self.__computeDrugMean()
        self.__computeMiRNAMean()
        self.__globalAverage()

    def __generateSet(self):
        scale = set()
        if self.evalSettings.contains('-val'):
            random.shuffle(self.trainingData)
            separation = int(self.elemCount()*float(self.evalSettings['-val']))
            self.testData = self.trainingData[:separation]
            self.trainingData = self.trainingData[separation:]
        for i,entry in enumerate(self.trainingData):
            miRNAName,drugName,rating = entry
            if miRNAName not in self.miRNA:
                self.miRNA[miRNAName] = len(self.miRNA)
                self.id2miRNA[self.miRNA[miRNAName]] = miRNAName
            if drugName not in self.drug:
                self.drug[drugName] = len(self.drug)
                self.id2drug[self.drug[drugName]] = drugName
            self.trainSet_u[miRNAName][drugName] = rating
            self.trainSet_i[drugName][miRNAName] = rating
            scale.add(float(rating))
        self.rScale = list(scale)
        self.rScale.sort()
        for entry in self.testData:
            if self.evalSettings.contains('-predict'):
                self.testSet_u[entry]={}
            else:
                miRNAName, drugName, rating = entry
                self.testSet_u[miRNAName][drugName] = rating
                self.testSet_i[drugName][miRNAName] = rating

    def __globalAverage(self):
        total = sum(self.miRNAMeans.values())
        if total==0:
            self.globalMean = 0
        else:
            self.globalMean = total/len(self.miRNAMeans)

    def __computeMiRNAMean(self):
        for u in self.miRNA:
            self.miRNAMeans[u] = sum(self.trainSet_u[u].values())/len(self.trainSet_u[u])

    def __computeDrugMean(self):
        for c in self.drug:
            self.drugMeans[c] = sum(self.trainSet_i[c].values())/len(self.trainSet_i[c])

    def trainingSize(self):
        return (len(self.miRNA),len(self.drug),len(self.trainingData))

    def testSize(self):
        return (len(self.testSet_u),len(self.testSet_i),len(self.testData))

class SparseMatrix():
    def __init__(self,triple):
        self.matrix_MiRNA = {}
        self.matrix_Drug = {}
        for drug in triple:
            if drug[0] not in self.matrix_MiRNA:
                self.matrix_MiRNA[drug[0]] = {}
            if drug[1] not in self.matrix_Drug:
                self.matrix_Drug[drug[1]] = {}
            self.matrix_MiRNA[drug[0]][drug[1]]=drug[2]
            self.matrix_Drug[drug[1]][drug[0]]=drug[2]
        self.elemNum = len(triple)
        self.size = (len(self.matrix_MiRNA),len(self.matrix_Drug))