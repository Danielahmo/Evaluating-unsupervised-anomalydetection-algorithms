# -*- coding: utf-8 -*-
"""Copia de Assignment 5

Daniela Herrera Montes de Oca
Pedro Esteban Chavarrias Solano

"""
import os
os.system('pip install brminer')
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import brminer
from scipy.io.arff import loadarff
from sklearn.mixture import GaussianMixture
import seaborn as sns

from scipy.stats import friedmanchisquare, wilcoxon
import itertools

"""## Dissimilarity measures"""

import numpy as np
import os
import random
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_distances, haversine_distances
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy.spatial.distance import cdist

class BRM(BaseEstimator):
    def __init__(self, classifier_count=100, bootstrap_sample_percent=100, use_bootstrap_sample_count=False,
                 bootstrap_sample_count=0, use_past_even_queue=False, max_event_count=3,
                 alpha=0.5, user_threshold=95, dissimilarity_measure='euclidean'):
        self.classifier_count = classifier_count
        self.bootstrap_sample_percent = bootstrap_sample_percent
        self.use_bootstrap_sample_count = use_bootstrap_sample_count
        self.bootstrap_sample_count = bootstrap_sample_count
        self.use_past_even_queue = use_past_even_queue
        self.max_event_count = max_event_count
        self.alpha = alpha
        self.user_threshold = user_threshold
        self.dissimilarity_measure = dissimilarity_measure

    def _evaluate(self, current_similarity):
        if (current_similarity < 0):
            current_similarity = 0

        if (self.use_past_even_queue == False):
            return -1 + 2 * current_similarity

        result_similarity = (
                    self.alpha * self._similarity_sum / self.max_event_count + (1 - self.alpha) * current_similarity)
        if (result_similarity < 0):
            result_similarity = 0

        self._similarity_sum += current_similarity

        if (len(self._past_events) == self.max_event_count):
            self._similarity_sum -= self._past_events.pop(0)

        self._past_events.append(current_similarity)

        if (self._similarity_sum < 0):
            self._similarity_sum = 0

        return -1 + 2 * result_similarity

    def dissimilarity(self, dissimilarity_measure, x, y):
        if dissimilarity_measure == 'euclidean':
            return euclidean_distances(x, y)
        if dissimilarity_measure == 'manhattan':
            return manhattan_distances(x, y)
        if dissimilarity_measure == 'minkowski':
            return cdist(x, y, metric='minkowski', p=0.5)
        if dissimilarity_measure == 'cosine':
            return cosine_distances(x, y)

    def score_samples(self, X):
        # distance = self.dissimilarity(self.dissimilarity_measure)
        X = np.array(X)
        X_test = self._scaler.transform(X)

        result = []
        batch_size = 100
        for i in range(min(len(X_test), batch_size), len(X_test) + batch_size, batch_size):
            current_X_test = X_test[[j for j in range(max(0, i - batch_size), min(i, len(X_test)))]]

            current_similarity = np.average([np.exp(-np.power(
                np.amin(self.dissimilarity(self.dissimilarity_measure, current_X_test, self._centers[i]),
                        axis=1) / self._max_dissimilarity, 2) / (self._sd[i])) for i in range(len(self._centers))],
                                            axis=0)

            result = result + [j for j in list(map(self._evaluate, current_similarity))]

        return result

    def predict(self, X):
        if (len(X.shape) < 2):
            raise ValueError('Reshape your data')

        if (X.shape[1] != self.n_features_in_):
            raise ValueError('Reshape your data')

        if not self._is_threshold_Computed:
            x_pred_classif = self.score_samples(self._X_train)
            x_pred_classif.sort()
            self._inner_threshold = x_pred_classif[(100 - self.user_threshold) * len(x_pred_classif) // 100]
            self._is_threshold_Computed = True

        y_pred_classif = self.score_samples(X)
        return [-1 if s <= self._inner_threshold else 1 for s in y_pred_classif]

    def fit(self, X, y=None):
        # Check that X and y have correct shape
        # distance = self.dissimilarity(self.dissimilarity_measure)
        if y is not None:
            X_train, y_train = check_X_y(X, y)
        else:
            X_train = check_array(X)

        self._similarity_sum = 0
        self._is_threshold_Computed = False

        self.n_features_in_ = X_train.shape[1]

        if self.n_features_in_ < 1:
            raise ValueError('Unable to instantiate the train dataset - Empty vector')

        self._scaler = MinMaxScaler()
        X_train = pd.DataFrame(X_train)
        X_train = pd.DataFrame(self._scaler.fit_transform(X_train[X_train.columns]), index=X_train.index,
                               columns=X_train.columns)

        self._max_dissimilarity = math.sqrt(self.n_features_in_)
        self._sd = np.empty(0)
        sampleSize = int(self.bootstrap_sample_count) if (self.use_bootstrap_sample_count) else int(
            0.01 * self.bootstrap_sample_percent * len(X_train));
        self._centers = np.empty((0, sampleSize, self.n_features_in_))

        list_instances = X_train.values.tolist()
        for i in range(0, self.classifier_count):
            centers = random.choices(list_instances, k=sampleSize)
            self._centers = np.insert(self._centers, i, centers, axis=0)
            self._sd = np.insert(self._sd, i, 2 * (np.mean(
                self.dissimilarity(self.dissimilarity_measure, centers, centers)) / self._max_dissimilarity) ** 2)

        return self

"""## Models """

# Function importing Dataset 
def importdata(trainFile, testFile): 
    test_arff = loadarff(testFile) 
    test = pd.DataFrame(test_arff[0])
    test['Class'] = test['Class'].str.decode('utf-8')

    train_arff = loadarff(trainFile) 
    train = pd.DataFrame(train_arff[0])
    train['Class'] = train['Class'].str.decode('utf-8')
    return train, test    

# Function to split target from data 
def splitdataset(train, test): 
    ohe = OneHotEncoder(sparse=True)
    objInTrain = len(train)

    allData = pd.concat([train, test], ignore_index=True, sort =False, axis=0)
    AllDataWihoutClass = allData.iloc[:, :-1]
    AllDataWihoutClassOnlyNominals = AllDataWihoutClass.select_dtypes(include=['object'])
    AllDataWihoutClassNoNominals = AllDataWihoutClass.select_dtypes(exclude=['object'])

    encAllDataWihoutClassNominals = ohe.fit_transform(AllDataWihoutClassOnlyNominals)
    encAllDataWihoutClassNominalsToPanda = pd.DataFrame(encAllDataWihoutClassNominals.toarray())
    
    if AllDataWihoutClassOnlyNominals.shape[1] > 0:
      codAllDataAgain = pd.concat([encAllDataWihoutClassNominalsToPanda, AllDataWihoutClassNoNominals], ignore_index=True, sort =False, axis=1)
    else:
      codAllDataAgain = AllDataWihoutClass

    # Seperating the target variable 
    X_train = codAllDataAgain[:objInTrain]
    y_train = train.values[:, -1]

    X_test = codAllDataAgain[objInTrain:]
    y_test = test.values[:, -1]
    
    mm_scaler = MinMaxScaler()
    X_train_minmax = pd.DataFrame(mm_scaler.fit_transform(X_train[X_train.columns]), index=X_train.index, columns=X_train.columns)
    X_test_minmax = pd.DataFrame(mm_scaler.transform(X_test[X_test.columns]), index=X_test.index, columns=X_test.columns)
    
    std_scaler = StandardScaler()
    X_train_std = pd.DataFrame(std_scaler.fit_transform(X_train[X_train.columns]), index=X_train.index, columns=X_train.columns)
    X_test_std = pd.DataFrame(std_scaler.transform(X_test[X_test.columns]), index=X_test.index, columns=X_test.columns)
    
    X_train_minmax_std = pd.DataFrame(std_scaler.fit_transform(X_train_minmax[X_train_minmax.columns]), index=X_train_minmax.index, columns=X_train_minmax.columns)
    X_test_minmax_std = pd.DataFrame(std_scaler.transform(X_test_minmax[X_test_minmax.columns]), index=X_test_minmax.index, columns=X_test_minmax.columns)
    
    return X_train, X_test, y_train, y_test, X_train_minmax, X_test_minmax, X_train_std, X_test_std, X_train_minmax_std, X_test_minmax_std

# Function to make predictions 
def prediction(X_test, clf_object):  
    y_pred = clf_object.score_samples(X_test) 
    return y_pred 

def result_of_Class(y_test, y_pred, saveFile):       
    np.savetxt(saveFile, y_pred, fmt='%.4f')

rootDir = 'Unsupervised_Anomaly_Detection/'
def Model_test(clf_classif, name_classifier):
  data = {'folder_name': [],
                  'auc': []}
  for dirName, subdirList, fileList in os.walk(rootDir):
      print('Directorio encontrado: %s' % dirName)
      print("************************************ DIRECTORIO **************************************")
      if len(fileList) > 0: 
          arr_auc = []
          arr_folder_name = dirName.split("/")
          folder_name = arr_folder_name[len(arr_folder_name) - 1]
          completed_name = folder_name + "-5-"
          for i in range(1, int(len(fileList) / 2) + 1):
              print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DATASET !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!") 
              trainFile = str(dirName) + '/' + completed_name + str(i) +"tra.dat"
              testFile = str(dirName) + '/' + completed_name + str(i) +"tst.dat"
              print('trainFile ' + trainFile)
              print('testFile ' + testFile)
              # Loading data 
              train, test = data2DF(trainFile), data2DF(testFile)

              # Training
              X_train, X_test, y_train, y_test, X_train_minmax, X_test_minmax, X_train_std, X_test_std, X_train_minmax_std, X_test_minmax_std = splitdataset(train, test)

              # Performing training 
              clf_classif.fit(X_train, y_train)

              # Operational Phase 
              y_pred_classif = prediction(X_test, clf_classif)

              auc = metrics.roc_auc_score(y_test,  y_pred_classif)
              arr_auc.append(1 - auc if auc < 0.5 else auc)
              print("AUC: "+str(1 - auc if auc < 0.5 else auc))

          print('AUC!! ' + str(arr_auc))
          aver_auc = sum(arr_auc) / len(arr_auc)
          print('aver_auc!! ' + str(aver_auc))
          data['folder_name'].append(folder_name)
          data['auc'].append(aver_auc)
          print('data[auc] ' + str(data['auc']))

      df = pd.DataFrame(data, columns = ['folder_name', 'auc'])    
      df.to_csv('occ_results_{}.csv'.format(name_classifier))

def data2DF (file_name):
  data = [i.strip().split('\n') for i in open(file_name).readlines()]
  c = [data.index(row) for row in data if '@data' in row]
  columns_c = data[c[0]-2][0].replace(',', '').split(' ')[1:]+ ['Class']
  columns = [i.replace(',','')for i in columns_c]
  data = np.array(data)[c[0]+1:]
  data = [i[0].split(',') for i in data]
  data_array = [np.array(i) for i in data]
  data_df = pd.DataFrame(data_array, columns = columns).replace(',','', regex=True)
  for col in columns:
    if data_df[col][0].replace('.', '').isdigit()==True:
      data_df[col] = data_df[col].astype(float)
  return data_df

Model_test(brminer.BRM(dissimilarity_measure = 'euclidean'),  'BRM_euclidean')

Model_test(brminer.BRM(dissimilarity_measure = 'manhattan'),  'BRM_manhattan')

Model_test(brminer.BRM(dissimilarity_measure = 'minkowski'),  'BRM_minkowski')

Model_test(brminer.BRM(dissimilarity_measure = 'cosine'),  'BRM_cosine')

Model_test(GaussianMixture(), 'GMM')

Model_test(IsolationForest(), 'ISOF')

Model_test(OneClassSVM(), 'ocSVM')

"""## Stadistic tests

### Boxplot
"""

GMM = pd.read_csv('occ_results_GMM.csv')
ISOF = pd.read_csv('occ_results_ISOF.csv')
ocSVM = pd.read_csv('occ_results_ocSVM.csv')
BRM = pd.read_csv('occ_results_BRM.csv')
BRM_manhattan = pd.read_csv('occ_results_BRM_manhattan.csv')
BRM_minkowski = pd.read_csv('occ_results_BRM_minkowski.csv')
BRM_cosine = pd.read_csv('occ_results_BRM_cosine.csv')

models_df = pd.concat([GMM['auc'],ISOF['auc']], axis = 1, names = ['GMM', 'ISOF']) 
models_df = pd.concat([models_df, ocSVM['auc']], axis = 1, names = ['GMM', 'ISOF', 'ocSVM'])
models_df = pd.concat([models_df, BRM['auc']], axis = 1, names = ['GMM', 'ISOF', 'ocSVM', 'BRM'])
models_df = pd.concat([models_df, BRM_manhattan['auc']], axis = 1, names = ['GMM', 'ISOF', 'ocSVM', 'BRM', 'BRM_manhattan'])
models_df = pd.concat([models_df, BRM_minkowski['auc']], axis = 1, names = ['GMM', 'ISOF', 'ocSVM', 'BRM', 'BRM_manhattan', 'BRM_minkowski'])
models_df = pd.concat([models_df, BRM_cosine['auc']], axis = 1, names = ['GMM', 'ISOF', 'ocSVM', 'BRM', 'BRM_manhattan', 'BRM_minkowski', 'BRM_cosine'])
models_df.columns = ['GMM', 'ISOF', 'ocSVM', 'BRM', 'BRM_manhattan', 'BRM_minkowski', 'BRM_cosine']


ax = sns.boxplot( data= models_df)
ax.set(xlabel= 'Models', ylabel='AUC')
ax.set_title('Results of algorithms with Standard normalizing')
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)

"""### Friedman test with a posthoc test

Friedman
"""

f_test = friedmanchisquare(models_df['GMM'], models_df['ISOF'], models_df['ocSVM'], models_df['BRM'], models_df['BRM_manhattan'], models_df['BRM_minkowski'],models_df['BRM_cosine'])
f_res = pd.DataFrame({'test':'Friedman','statistic':f_test[0],'pvalue':f_test[1]},index=[0])

"""Pairwise Wilcoxon"""

wilc_test = [wilcoxon(models_df[i],models_df[j]) for i,j in itertools.combinations(models_df.columns,2)]    
w_res = pd.DataFrame(wilc_test)
w_res['test'] = ["wilcoxon " + i+" vs "+j for i,j in itertools.combinations(models_df.columns,2)]

print(f_test)

print(pd.concat([f_res,w_res]))

BRM = pd.read_csv('occ_results_BRM.csv', index_col = 0)
BRM_manhattan = pd.read_csv('occ_results_BRM_manhattan.csv', index_col = 0)
BRM_minkowski = pd.read_csv('occ_results_BRM_minkowski.csv', index_col = 0)
BRM_cosine = pd.read_csv('occ_results_BRM_cosine.csv', index_col = 0)
ISOF = pd.read_csv('occ_results_ISOF.csv',index_col = 0)
GMM = pd.read_csv('occ_results_GMM.csv', index_col = 0)
OCSVM = pd.read_csv('occ_results_ocSVM.csv',index_col= 0)

BRM['Classifier'] = ['BRM'] * 60
BRM_manhattan['Classifier'] = ['BRM_manhattan'] * 60
BRM_minkowski['Classifier'] = ['BRM_minkowski'] * 60
BRM_cosine['Classifier'] = ['BRM_cosine'] * 60
ISOF['Classifier'] = ['ISOF'] * 60
GMM['Classifier'] = ['GMM'] * 60
OCSVM['Classifier'] = ['ocSVM'] * 60

cd_df = pd.DataFrame(BRM)
cd_df = pd.concat([cd_df,BRM_manhattan])
cd_df = pd.concat([cd_df,BRM_minkowski])
cd_df = pd.concat([cd_df,BRM_cosine])
cd_df = pd.concat([cd_df,ISOF])
cd_df = pd.concat([cd_df,GMM])
cd_df = pd.concat([cd_df,OCSVM])
cd_df.to_csv('CD_DF.csv')

CD = pd.read_csv('CD_DF.csv', index_col=0).rename(columns = {'folder_name': 'dataset_name', 'Classifier' :'classifier_name', 'auc':'accuracy'})

##############################################################################################################
####################################### CRITICAL DIFFERENCE DIAGRAM ##########################################
"""Program obtained from https://github.com/hfawaz/cd-diagram for computing the critical difference diagram"""

import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'

import operator
import math
from scipy.stats import wilcoxon
from scipy.stats import friedmanchisquare
import networkx

# inspired from orange3 https://docs.orange.biolab.si/3/data-mining-library/reference/evaluation.cd.html
def graph_ranks(avranks, names, p_values, cd=None, cdmethod=None, lowv=None, highv=None,
                width=6, textspace=1, reverse=False, filename=None, labels=False, **kwargs):
    """
    Draws a CD graph, which is used to display  the differences in methods'
    performance. See Janez Demsar, Statistical Comparisons of Classifiers over
    Multiple Data Sets, 7(Jan):1--30, 2006.
    Needs matplotlib to work.
    The image is ploted on `plt` imported using
    `import matplotlib.pyplot as plt`.
    Args:
        avranks (list of float): average ranks of methods.
        names (list of str): names of methods.
        cd (float): Critical difference used for statistically significance of
            difference between methods.
        cdmethod (int, optional): the method that is compared with other methods
            If omitted, show pairwise comparison of methods
        lowv (int, optional): the lowest shown rank
        highv (int, optional): the highest shown rank
        width (int, optional): default width in inches (default: 6)
        textspace (int, optional): space on figure sides (in inches) for the
            method names (default: 1)
        reverse (bool, optional):  if set to `True`, the lowest rank is on the
            right (default: `False`)
        filename (str, optional): output file name (with extension). If not
            given, the function does not write a file.
        labels (bool, optional): if set to `True`, the calculated avg rank
        values will be displayed
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError:
        raise ImportError("Function graph_ranks requires matplotlib.")

    width = float(width)
    textspace = float(textspace)

    def nth(l, n):
        """
        Returns only nth elemnt in a list.
        """
        n = lloc(l, n)
        return [a[n] for a in l]

    def lloc(l, n):
        """
        List location in list of list structure.
        Enable the use of negative locations:
        -1 is the last element, -2 second last...
        """
        if n < 0:
            return len(l[0]) + n
        else:
            return n

    def mxrange(lr):
        """
        Multiple xranges. Can be used to traverse matrices.
        This function is very slow due to unknown number of
        parameters.
        >>> mxrange([3,5])
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        >>> mxrange([[3,5,1],[9,0,-3]])
        [(3, 9), (3, 6), (3, 3), (4, 9), (4, 6), (4, 3)]
        """
        if not len(lr):
            yield ()
        else:
            # it can work with single numbers
            index = lr[0]
            if isinstance(index, int):
                index = [index]
            for a in range(*index):
                for b in mxrange(lr[1:]):
                    yield tuple([a] + list(b))

    def print_figure(fig, *args, **kwargs):
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(*args, **kwargs)

    sums = avranks

    nnames = names
    ssums = sums

    if lowv is None:
        lowv = min(1, int(math.floor(min(ssums))))
    if highv is None:
        highv = max(len(avranks), int(math.ceil(max(ssums))))

    cline = 0.4

    k = len(sums)

    lines = None

    linesblank = 0
    scalewidth = width - 2 * textspace

    def rankpos(rank):
        if not reverse:
            a = rank - lowv
        else:
            a = highv - rank
        return textspace + scalewidth / (highv - lowv) * a

    distanceh = 0.25

    cline += distanceh

    # calculate height needed height of an image
    minnotsignificant = max(2 * 0.2, linesblank)
    height = cline + ((k + 1) / 2) * 0.2 + minnotsignificant

    fig = plt.figure(figsize=(width, height))
    fig.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1])  # reverse y axis
    ax.set_axis_off()

    hf = 1. / height  # height factor
    wf = 1. / width

    def hfl(l):
        return [a * hf for a in l]

    def wfl(l):
        return [a * wf for a in l]

    # Upper left corner is (0,0).
    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    def line(l, color='k', **kwargs):
        """
        Input is a list of pairs of points.
        """
        ax.plot(wfl(nth(l, 0)), hfl(nth(l, 1)), color=color, **kwargs)

    def text(x, y, s, *args, **kwargs):
        ax.text(wf * x, hf * y, s, *args, **kwargs)

    line([(textspace, cline), (width - textspace, cline)], linewidth=2)

    bigtick = 0.3
    smalltick = 0.15
    linewidth = 2.0
    linewidth_sign = 4.0

    tick = None
    for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick
        if a == int(a):
            tick = bigtick
        line([(rankpos(a), cline - tick / 2),
              (rankpos(a), cline)],
             linewidth=2)

    for a in range(lowv, highv + 1):
        text(rankpos(a), cline - tick / 2 - 0.05, str(a),
             ha="center", va="bottom", size=16)

    k = len(ssums)

    def filter_names(name):
        return name

    space_between_names = 0.24

    for i in range(math.ceil(k / 2)):
        chei = cline + minnotsignificant + i * space_between_names
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace - 0.1, chei)],
             linewidth=linewidth)
        if labels:
            text(textspace + 0.3, chei - 0.075, format(ssums[i], '.4f'), ha="right", va="center", size=10)
        text(textspace - 0.2, chei, filter_names(nnames[i]), ha="right", va="center", size=16)

    for i in range(math.ceil(k / 2), k):
        chei = cline + minnotsignificant + (k - i - 1) * space_between_names
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace + scalewidth + 0.1, chei)],
             linewidth=linewidth)
        if labels:
            text(textspace + scalewidth - 0.3, chei - 0.075, format(ssums[i], '.4f'), ha="left", va="center", size=10)
        text(textspace + scalewidth + 0.2, chei, filter_names(nnames[i]),
             ha="left", va="center", size=16)

    # no-significance lines
    def draw_lines(lines, side=0.05, height=0.1):
        start = cline + 0.2

        for l, r in lines:
            line([(rankpos(ssums[l]) - side, start),
                  (rankpos(ssums[r]) + side, start)],
                 linewidth=linewidth_sign)
            start += height
            print('drawing: ', l, r)

    # draw_lines(lines)
    start = cline + 0.2
    side = -0.02
    height = 0.1

    # draw no significant lines
    # get the cliques
    cliques = form_cliques(p_values, nnames)
    i = 1
    achieved_half = False
    print(nnames)
    for clq in cliques:
        if len(clq) == 1:
            continue
        print(clq)
        min_idx = np.array(clq).min()
        max_idx = np.array(clq).max()
        if min_idx >= len(nnames) / 2 and achieved_half == False:
            start = cline + 0.25
            achieved_half = True
        line([(rankpos(ssums[min_idx]) - side, start),
              (rankpos(ssums[max_idx]) + side, start)],
             linewidth=linewidth_sign)
        start += height


def form_cliques(p_values, nnames):
    """
    This method forms the cliques
    """
    # first form the numpy matrix data
    m = len(nnames)
    g_data = np.zeros((m, m), dtype=np.int64)
    for p in p_values:
        if p[3] == False:
            i = np.where(nnames == p[0])[0][0]
            j = np.where(nnames == p[1])[0][0]
            min_i = min(i, j)
            max_j = max(i, j)
            g_data[min_i, max_j] = 1

    g = networkx.Graph(g_data)
    return networkx.find_cliques(g)


def draw_cd_diagram(df_perf=None, alpha=0.05, title=None, labels=False):
    """
    Draws the critical difference diagram given the list of pairwise classifiers that are
    significant or not
    """
    p_values, average_ranks, _ = wilcoxon_holm(df_perf=df_perf, alpha=alpha)

    print(average_ranks)

    for p in p_values:
        print(p)


    graph_ranks(average_ranks.values, average_ranks.keys(), p_values,
                cd=None, reverse=True, width=9, textspace=1.5, labels=labels)

    font = {'family': 'sans-serif',
        'color':  'black',
        'weight': 'normal',
        'size': 22,
        }
    if title:
        plt.title(title,fontdict=font, y=0.9, x=0.5)
    plt.savefig('cd-diagram.png',bbox_inches='tight')

def wilcoxon_holm(alpha=0.05, df_perf=None):
    """
    Applies the wilcoxon signed rank test between each pair of algorithm and then use Holm
    to reject the null's hypothesis
    """
    print(pd.unique(df_perf['classifier_name']))
    # count the number of tested datasets per classifier
    df_counts = pd.DataFrame({'count': df_perf.groupby(
        ['classifier_name']).size()}).reset_index()
    # get the maximum number of tested datasets
    max_nb_datasets = df_counts['count'].max()
    # get the list of classifiers who have been tested on nb_max_datasets
    classifiers = list(df_counts.loc[df_counts['count'] == max_nb_datasets]
                       ['classifier_name'])
    # test the null hypothesis using friedman before doing a post-hoc analysis
    friedman_p_value = friedmanchisquare(*(
        np.array(df_perf.loc[df_perf['classifier_name'] == c]['accuracy'])
        for c in classifiers))[1]
    if friedman_p_value >= alpha:
        # then the null hypothesis over the entire classifiers cannot be rejected
        print('the null hypothesis over the entire classifiers cannot be rejected')
        exit()
    # get the number of classifiers
    m = len(classifiers)
    # init array that contains the p-values calculated by the Wilcoxon signed rank test
    p_values = []
    # loop through the algorithms to compare pairwise
    for i in range(m - 1):
        # get the name of classifier one
        classifier_1 = classifiers[i]
        # get the performance of classifier one
        perf_1 = np.array(df_perf.loc[df_perf['classifier_name'] == classifier_1]['accuracy']
                          , dtype=np.float64)
        for j in range(i + 1, m):
            # get the name of the second classifier
            classifier_2 = classifiers[j]
            # get the performance of classifier one
            perf_2 = np.array(df_perf.loc[df_perf['classifier_name'] == classifier_2]
                              ['accuracy'], dtype=np.float64)
            # calculate the p_value
            p_value = wilcoxon(perf_1, perf_2, zero_method='pratt')[1]
            # appen to the list
            p_values.append((classifier_1, classifier_2, p_value, False))
    # get the number of hypothesis
    k = len(p_values)
    # sort the list in acsending manner of p-value
    p_values.sort(key=operator.itemgetter(2))

    # loop through the hypothesis
    for i in range(k):
        # correct alpha with holm
        new_alpha = float(alpha / (k - i))
        # test if significant after holm's correction of alpha
        if p_values[i][2] <= new_alpha:
            p_values[i] = (p_values[i][0], p_values[i][1], p_values[i][2], True)
        else:
            # stop
            break
    # compute the average ranks to be returned (useful for drawing the cd diagram)
    # sort the dataframe of performances
    sorted_df_perf = df_perf.loc[df_perf['classifier_name'].isin(classifiers)]. \
        sort_values(['classifier_name', 'dataset_name'])
    # get the rank data
    rank_data = np.array(sorted_df_perf['accuracy']).reshape(m, max_nb_datasets)

    # create the data frame containg the accuracies
    df_ranks = pd.DataFrame(data=rank_data, index=np.sort(classifiers), columns=
    np.unique(sorted_df_perf['dataset_name']))

    # number of wins
    dfff = df_ranks.rank(ascending=False)
    print(dfff[dfff == 1.0].sum(axis=1))

    # average the ranks
    average_ranks = df_ranks.rank(ascending=False).mean(axis=1).sort_values(ascending=False)
    # return the p-values and the average ranks
    return p_values, average_ranks, max_nb_datasets

draw_cd_diagram(df_perf=CD, title='AUC without Minmax Scaling and Standard Normalizing', labels=True)
