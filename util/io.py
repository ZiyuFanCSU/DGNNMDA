from re import split
from .config import LineConfig

class FileIO(object):
    def __init__(self):
        pass

    @staticmethod
    def loadDataSet(conf, file, bTest=False):
        trainingData = []
        testData = []
        ratingConfig = LineConfig(conf['ratings.setup'])
        with open(file) as f:
            ratings = f.readlines()
        if ratingConfig.contains('-header'):
            ratings = ratings[1:]
        order = ratingConfig['-columns'].strip().split()
        delim = ' |,|\t'
        for lineNo, line in enumerate(ratings):
            drugs = split(delim,line.strip())
            miRNAId = drugs[int(order[0])]
            drugId = drugs[int(order[1])]
            rating = 1
            if bTest:
                testData.append([miRNAId, drugId, float(rating)])
            else:
                trainingData.append([miRNAId, drugId, float(rating)])
        if bTest:
            return testData
        else:
            return trainingData

    @staticmethod
    def loadRelationship(conf, filePath):
        socialConfig = LineConfig(conf['miRNA.setup'])
        relation = []
        with open(filePath) as f:
            relations = f.readlines()
        if socialConfig.contains('-header'):
            relations = relations[1:]
        order = socialConfig['-columns'].strip().split()
        for lineNo, line in enumerate(relations):
            drugs = split(' |,|\t', line.strip())
            miRNAId1 = drugs[int(order[0])]
            miRNAId2 = drugs[int(order[1])]
            if len(order) < 3:
                weight = 1
            else:
                weight = float(drugs[int(order[2])])
            relation.append([miRNAId1, miRNAId2, weight])
        return relation