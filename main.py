import tensorflow as tf
from scipy.sparse import coo_matrix
from util.config import Config
from random import shuffle, choice
from util.io import FileIO
from numpy import *
import os
from util import config
import numpy as np
from util.config import LineConfig
from util.Tool import SparseMatrix,Rating
from collections import defaultdict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

class DGNNMDA:
    def __init__(self, conf, trainingSet=None, testSet=None, relation=None):
        self.config = conf
        self.miRNA = {}  # used to store the order of miRNAs
        self.relation = relation
        self.followees = defaultdict(dict)
        self.followers = defaultdict(dict)
        self.trustMatrix = self.__generateSet()
        self.data = Rating(self.config, trainingSet, testSet)
        self.dataClean()
        self.evalSettings = LineConfig(self.config['evaluation.setup'])
        self.num_miRNAs, self.num_drugs, self.train_size = self.data.trainingSize()
        self.emb_size = int(self.config['num.factors'])
        self.maxIter = int(self.config['num.max.iter'])
        learningRate = config.LineConfig(self.config['learnRate'])
        self.lRate = float(learningRate['-init'])
        self.batch_size = int(self.config['batch_size'])
        regular = config.LineConfig(self.config['reg.lambda'])
        self.regU = float(regular['-u'])
        args = config.LineConfig(self.config['DGNNMDA'])
        self.n_layers = int(args['-n_layer'])
        args2 = config.LineConfig(self.config['DGNNMDA2'])
        self.n_layers2 = int(args2['-n_layer'])

    def dataClean(self):
        cleanList = []
        cleanPair = []
        for miRNA in self.followees:
            if miRNA not in self.data.miRNA:
                cleanList.append(miRNA)
            for u2 in self.followees[miRNA]:
                if u2 not in self.data.miRNA:
                    cleanPair.append((miRNA, u2))
        for u in cleanList:
            del self.followees[u]
        for pair in cleanPair:
            if pair[0] in self.followees:
                del self.followees[pair[0]][pair[1]]
        cleanList = []
        cleanPair = []
        for miRNA in self.followers:
            if miRNA not in self.data.miRNA:
                cleanList.append(miRNA)
            for u2 in self.followers[miRNA]:
                if u2 not in self.data.miRNA:
                    cleanPair.append((miRNA, u2))
        for u in cleanList:
            del self.followers[u]
        for pair in cleanPair:
            if pair[0] in self.followers:
                del self.followers[pair[0]][pair[1]]
        idx = []
        for n, pair in enumerate(self.relation):
            if pair[0] not in self.data.miRNA or pair[1] not in self.data.miRNA:
                idx.append(n)
        for drug in reversed(idx):
            del self.relation[drug]

    def __generateSet(self):
        triple = []
        for line in self.relation:
            miRNAId1, miRNAId2, weight = line
            self.followees[miRNAId1][miRNAId2] = weight
            self.followers[miRNAId2][miRNAId1] = weight
            if miRNAId1 not in self.miRNA:
                self.miRNA[miRNAId1] = len(self.miRNA)
            if miRNAId2 not in self.miRNA:
                self.miRNA[miRNAId2] = len(self.miRNA)
            triple.append([self.miRNA[miRNAId1], self.miRNA[miRNAId2], weight])
        return SparseMatrix(triple)

    def next_batch_pairwise(self):
        shuffle(self.data.trainingData)
        batch_id = 0
        while batch_id < self.train_size:
            if batch_id + self.batch_size <= self.train_size:
                miRNAs = [self.data.trainingData[idx][0] for idx in range(batch_id, self.batch_size + batch_id)]
                drugs = [self.data.trainingData[idx][1] for idx in range(batch_id, self.batch_size + batch_id)]
                batch_id += self.batch_size
            else:
                miRNAs = [self.data.trainingData[idx][0] for idx in range(batch_id, self.train_size)]
                drugs = [self.data.trainingData[idx][1] for idx in range(batch_id, self.train_size)]
                batch_id = self.train_size

            u_idx, i_idx, j_idx = [], [], []
            drug_list = list(self.data.drug.keys())
            for i, miRNA in enumerate(miRNAs):
                i_idx.append(self.data.drug[drugs[i]])
                u_idx.append(self.data.miRNA[miRNA])
                neg_drug = choice(drug_list)
                while neg_drug in self.data.trainSet_u[miRNA]:
                    neg_drug = choice(drug_list)
                j_idx.append(self.data.drug[neg_drug])

            yield u_idx, i_idx, j_idx

    def buildSparseRelationMatrix(self):
        row, col, entries = [], [], []
        for pair in self.relation:
            row += [self.data.miRNA[pair[0]]]
            col += [self.data.miRNA[pair[1]]]
            entries += [1.0 / len(self.followees[pair[0]])]
        AdjacencyMatrix = coo_matrix((entries, (row, col)), shape=(self.num_miRNAs, self.num_miRNAs), dtype=np.float32)
        return AdjacencyMatrix

    def buildSparseRelationMatrix2(self):
        row, col, entries = [], [], []
        L = []
        with open("././dataset/d-d.txt", 'r') as f:
            L = f.readlines()
            L = [i.rstrip().split('\t') for i in L]
            for i in L:
                i[2] = 1
        l1 = mat(L)[:, 0:1].T.tolist()[0]
        l2 = mat(L)[:, 1:2].T.tolist()[0]
        z = {}
        for i, j in zip(l1, l2):
            if i not in z.keys():
                z[i] = j
            else:
                if isinstance(z[i], list):
                    z[i].append(j)
                else:
                    z[i] = [z[i]]
                    z[i].append(j)
        dic = {}
        for j in range(0, 145):
            if str(j) in z.keys():
                for i in z[str(j)]:
                    dic.setdefault(str(j), {})[i] = 1
            else:
                dic.setdefault(str(j), {})
        for pair in L:
            if pair[0] in self.data.drug.keys():
                if pair[1] in self.data.drug.keys():
                    row += [self.data.drug[pair[0]]]
                    col += [self.data.drug[pair[1]]]
                    entries += [1.0 / len(dic[pair[0]])]
        AdjacencyMatrix = coo_matrix((entries, (row, col)), shape=(self.num_drugs, self.num_drugs), dtype=np.float32)
        return AdjacencyMatrix

    def buildSparseRatingMatrix(self):
        row, col, entries = [], [], []
        for pair in self.data.trainingData:
            row += [self.data.miRNA[pair[0]]]
            col += [self.data.drug[pair[1]]]
            entries += [1.0 / len(self.data.trainSet_u[pair[0]])]
        ratingMatrix = coo_matrix((entries, (row, col)), shape=(self.num_miRNAs, self.num_drugs), dtype=np.float32)
        return ratingMatrix

    def buildSparseRatingMatrix2(self):
        row, col, entries = [], [], []
        for pair in self.data.trainingData:
            row += [self.data.drug[pair[1]]]
            col += [self.data.miRNA[pair[0]]]

            entries += [1.0 / len(self.data.trainSet_u[pair[0]])]
        ratingMatrix = coo_matrix((entries, (row, col)), shape=(self.num_drugs, self.num_miRNAs), dtype=np.float32)
        return ratingMatrix

    def initModel(self):
        self.u_idx = tf.placeholder(tf.int32, name="u_idx")
        self.v_idx = tf.placeholder(tf.int32, name="v_idx")
        self.r = tf.placeholder(tf.float32, name="rating")

        num = self.num_miRNAs + self.num_drugs
        list_three = [[0 for i in range(num)] for j in range(num)]
        for i in range(num):
            list_three[i][i] = 1
        from sklearn.decomposition import PCA
        pca = PCA(n_components=32)
        a = pca.fit_transform(list_three)
        a = a.astype(np.float32)
        a = tf.convert_to_tensor(a)
        self.miRNA_embeddings = a[:self.num_miRNAs]
        self.drug_embeddings = a[self.num_miRNAs:]

        self.u_embedding = tf.nn.embedding_lookup(self.miRNA_embeddings, self.u_idx)
        self.v_embedding = tf.nn.embedding_lookup(self.drug_embeddings, self.v_idx)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        S = self.buildSparseRelationMatrix()
        S2 = self.buildSparseRelationMatrix2()
        C = self.buildSparseRatingMatrix()
        D = self.buildSparseRatingMatrix2()
        indices = np.mat([S.row, S.col]).transpose()
        self.S = tf.SparseTensor(indices, S.data.astype(np.float32), S.shape)

        indices2 = np.mat([S2.row, S2.col]).transpose()
        self.S2 = tf.SparseTensor(indices2, S2.data.astype(np.float32), S2.shape)

        indices5 = np.mat([C.row, C.col]).transpose()
        self.C = tf.SparseTensor(indices5, C.data.astype(np.float32), C.shape)

        indices6 = np.mat([D.row, D.col]).transpose()
        self.D = tf.SparseTensor(indices6, D.data.astype(np.float32), D.shape)


    def buildModel(self):
        self.weights = {}
        self.weights2 = {}
        self.weights3 = {}
        self.weights4 = {}

        initializer = tf.contrib.layers.variance_scaling_initializer()
        initializer4 = tf.contrib.layers.variance_scaling_initializer()
        initializer2 = tf.contrib.layers.variance_scaling_initializer()
        initializer3 = tf.contrib.layers.variance_scaling_initializer()

        miRNA_embeddings00 = self.miRNA_embeddings
        drug_embeddings00 = self.drug_embeddings

        # miRNA的GraphSage
        for k in range(self.n_layers):
            self.weights['weights%d' % k] = tf.Variable(
                initializer([2 * self.emb_size, self.emb_size]), name='weights%d' % k)

        miRNA_embeddings = miRNA_embeddings00
        for k in range(self.n_layers):
            new_miRNA_embeddings = tf.sparse_tensor_dense_matmul(self.S, miRNA_embeddings)
            miRNA_embeddings = tf.matmul(tf.concat([new_miRNA_embeddings, miRNA_embeddings], 1),
                                         self.weights['weights%d' % k])
            miRNA_embeddings = tf.nn.leaky_relu(miRNA_embeddings)

        # drug的GraphSage
        for k in range(self.n_layers2):
            self.weights2['weights2%d' % k] = tf.Variable(
                initializer4([2 * self.emb_size, self.emb_size]), name='weights2%d' % k)

        drug_embeddings = drug_embeddings00
        for k in range(self.n_layers2):
            new_drug_embeddings = tf.sparse_tensor_dense_matmul(self.S2, drug_embeddings)
            drug_embeddings = tf.matmul(tf.concat([new_drug_embeddings, drug_embeddings], 1),
                                        self.weights2['weights2%d' % k])
            drug_embeddings = tf.nn.leaky_relu(drug_embeddings)

        # 
        k = 0
        self.weights3['weights3%d' % k] = tf.Variable(initializer2([2 * self.emb_size, self.emb_size]),
                                                      name='weights3%d' % k)
        middleResult_1 = 0.5 * tf.sparse_tensor_dense_matmul(self.C, (
                drug_embeddings + tf.multiply((tf.sparse_tensor_dense_matmul(self.D, miRNA_embeddings)),
                                              drug_embeddings)))
        miRNA_embeddings = tf.matmul(tf.concat([miRNA_embeddings, middleResult_1], 1), self.weights3['weights3%d' % k])
        final_miRNA_embeddings = tf.nn.leaky_relu(miRNA_embeddings)

        self.weights4['weights4%d' % k] = tf.Variable(initializer3([2 * self.emb_size, self.emb_size]),
                                                      name='weights4%d' % k)
        middleResult_2 = 0.5 * tf.sparse_tensor_dense_matmul(self.D, (
                miRNA_embeddings + tf.multiply((tf.sparse_tensor_dense_matmul(self.C, drug_embeddings)),
                                              miRNA_embeddings)))
        drug_embeddings = tf.matmul(tf.concat([drug_embeddings, middleResult_2], 1), self.weights4['weights4%d' % k])
        final_drug_embeddings = tf.nn.leaky_relu(drug_embeddings)


        self.final_miRNA = final_miRNA_embeddings
        self.final_drug = final_drug_embeddings

        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        self.neg_drug_embedding = tf.nn.embedding_lookup(final_drug_embeddings, self.neg_idx)
        self.u_embedding = tf.nn.embedding_lookup(final_miRNA_embeddings, self.u_idx)
        self.v_embedding = tf.nn.embedding_lookup(final_drug_embeddings, self.v_idx)
        self.test = tf.reduce_sum(tf.multiply(self.u_embedding, final_drug_embeddings), 1)

        y = tf.reduce_sum(tf.multiply(self.u_embedding, self.v_embedding), 1) - tf.reduce_sum(
            tf.multiply(self.u_embedding, self.neg_drug_embedding), 1)
        loss = -tf.reduce_sum(tf.log(tf.sigmoid(y))) + self.regU * (
                tf.nn.l2_loss(self.weights['weights%d' % 1]) + tf.nn.l2_loss(self.weights2['weights2%d' % 1])
                + tf.nn.l2_loss(self.weights['weights%d' % 0]) + tf.nn.l2_loss(self.weights2['weights2%d' % 0])
                + tf.nn.l2_loss(self.weights4['weights4%d' % 0]) + tf.nn.l2_loss(self.weights3['weights3%d' % 0]))
        opt = tf.train.AdamOptimizer(self.lRate)
        train = opt.minimize(loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)

        for iteration in range(self.maxIter):
            for n, batch in enumerate(self.next_batch_pairwise()):
                miRNA_idx, i_idx, j_idx = batch
                _, l = self.sess.run([train, loss],
                                     feed_dict={self.u_idx: miRNA_idx, self.neg_idx: j_idx, self.v_idx: i_idx})
                print('training:', iteration + 1, 'batch', n, 'loss:', l)


if __name__ == '__main__':
    config1 = Config('DGNNMDA.conf')
    i = 2
    train_path = f"./dataset/train.txt"
    test_path = f"./dataset/test.txt"
    train = FileIO.loadDataSet(config1, train_path)
    test = FileIO.loadDataSet(config1, test_path, bTest=True)
    rela = FileIO.loadRelationship(config1, config1['relation'])
    model = DGNNMDA(config1, train, test, rela)
    model.initModel()
    model.buildModel()
