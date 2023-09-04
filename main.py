import pickle
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import svm
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
# accuracy_score
from sklearn.metrics import confusion_matrix
# load dataset
file_dataset = 'dataset_tweet_sentiment_pilkada_DKI_2017.csv'
data = pd.read_csv(file_dataset)
# cetak informasi dataset
print(data.head())
print(data.info())

print('\nJumlah Data Berdasarkan Pasangan Calon:')
print(data.groupby('Pasangan Calon').size())

print('\nJumlah Data Sentiment Positive:')
dt = data.query("Sentiment == 'positive'")
print(dt.groupby('Pasangan Calon').size())

print('\nJumlah Data Sentiment Negative:')
dt = data.query("Sentiment == 'negative'")
print(dt.groupby('Pasangan Calon').size())

data['Sentiment'].value_counts().plot(kind='bar')
plt.show()

### KODE PROGRAM
def remove_at_hash(sent):
    return re.sub(r'@|#', r'', sent.lower())

def remove_sites(sent):
    return re.sub(r'http.*', r'', sent.lower())

def remove_punct(sent):
    return ' '.join(re.findall(r'\w+', sent.lower()))

data['text'] = data['Text Tweet'].apply(lambda x: remove_punct(remove_sites(remove_at_hash(x))))
print(data.head())

le = preprocessing.LabelEncoder()
le.fit(data['Sentiment'])
data['label'] = le.transform(data['Sentiment'])
print(data)

# TUGAS 1 : Nilai gamma SVM
pKernel = ['linear', 'rbf']
pC = [0.1, 1.0, 10.0]
# Menggunakan Nilai Gamma
pGamma = [0.1, 1.0, 10.0]
ik = 0
ic = 1
ig = 2
fs = False
# kernel SVM
# nilai C (hyperplane)
# indeks untuk kernel
# indeks untuk nilai C
# seleksi fitur, False=None, True=Chi-Square
print(f'Parameter SVM: Kernel={pKernel[ik]}, C={pC[ic]}, Gamma={pGamma[ig]}')
