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
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

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


# Pembersihan Data
def remove_at_hash(sent):
    return re.sub(r'@|#', r'', sent.lower())


def remove_sites(sent):
    return re.sub(r'http.*', r'', sent.lower())


def remove_punct(sent):
    return ' '.join(re.findall(r'\w+', sent.lower()))


data['text'] = data['Text Tweet'].apply(lambda x: remove_punct(remove_sites(remove_at_hash(x))))
print(data.head())

# Label Encoder
le = preprocessing.LabelEncoder()
le.fit(data['Sentiment'])
data['label'] = le.transform(data['Sentiment'])
print(data)

# Tugas 2 : Metode ekstraksi fitur selain TF-IDF (SPLIT DATA & TF-IDF)
X = data['text']
y = data['label']

# split data (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)
print(X_train[0], '-', y_train[0])
# ubah teks ke vektor dengan TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_train_vectors = tfidf_vectorizer.fit_transform(X_train)
tfidf_test_vectors = tfidf_vectorizer.transform(X_test)
print(tfidf_train_vectors[0])

# feature extraction menggunakan n-grams
ngram_vectorizer = CountVectorizer(ngram_range=(1, 2)) # unigram dan bigram
ngram_train_matrix = ngram_vectorizer.fit_transform(X_train)
ngram_test_matrix = ngram_vectorizer.transform(X_test)
feature_names = ngram_vectorizer.get_feature_names_out()
print(ngram_train_matrix.toarray())

# TUGAS 1 : Nilai gamma SVM (Kontrol Parameter SVM)
pKernel = ['linear', 'rbf']  # kernel SVM
pC = [0.1, 1.0, 10.0]  # nilai C (hyperplane)
pGamma = [0.1, 1.0, 10.0] # Nilai Gamma
pFitur = ['chisquare','randomforest','none']
ik = 0  # indeks untuk kernel
ic = 1 # indeks untuk nilai C
ig = 1 # indeks untuk nilai Gamma
fs = 2  # seleksi fitur
print(f'Parameter SVM: Kernel={pKernel[ik]}, C={pC[ic]}, Gamma={pGamma[ig]}')

# TUGAS 3 : Metode seleksi fitur selain Chi-Squar (Pemilihan Seleksi Fitur)
if pFitur[fs] == 'chisquare':
    fs_label = "ChiSquare"
    ch2 = SelectKBest(chi2, k=500)
    tfidf_train_vectors = ch2.fit_transform(tfidf_train_vectors, y_train)
    tfidf_test_vectors = ch2.transform(tfidf_test_vectors)
elif pFitur[fs] == 'randomforest':
    fs_label = "RandomForest"
    RF = RandomForestClassifier(n_estimators=500, max_features="sqrt", random_state=42)
    RF.fit(tfidf_train_vectors, y_train)
    feature_importances = RF.feature_importances_
    sfm = SelectFromModel(RF, threshold='median')
    tfidf_train_vectors = sfm.fit_transform(tfidf_train_vectors, y_train)
    tfidf_test_vectors = sfm.transform(tfidf_test_vectors)
else:
    fs_label = "None"

print(f'Seleksi Fitur SVM: {fs_label} dan Kernel {pKernel[ik]}')

# Training dan Testing SVM (80:20)
svm_classifier = svm.SVC(kernel=pKernel[ik], C=pC[ic], gamma=pGamma[ig])  # kernel={linear, rbf}, C={0.1,1.0,10.0}, gamma=[0.1,1.0,10.0]
svm_classifier.fit(tfidf_train_vectors, y_train)  # training
y_pred = svm_classifier.predict(tfidf_test_vectors)  # testing

print(classification_report(y_test, y_pred))

cnf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix (TN, FP, FN, TP):')
print(cnf_matrix)

# Visualisasi Confusion Matrix
group_names = ['TN', 'FP', 'FN', 'TP']
group_counts = ["{0:0.0f}".format(value) for value in cnf_matrix.flatten()]
labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
labels = np.asarray(labels).reshape(2, 2)
sns.heatmap(cnf_matrix, annot=labels, fmt='', cmap='Blues')
plt.show()


print('Statistik Hasil Data Dan Fitur Menggunakan TF-IDF : ')
print(f'Sel. Fitur\t: {fs_label}')
print(f'Param. SVM\t: Kernel={pKernel[ik]}, C={pC[ic]}, gamma={pGamma[ig]}')
# Statistik Hasil Data Dan Fitur Menggunakan TF-IDF
print(f'Jml. Data\t: {tfidf_train_vectors.shape[0]} (80%)')
print(f'Jml. Fitur\t: {tfidf_train_vectors.shape[1]}')

print('Precision\t: {:.2}'.format(precision_score(y_test, y_pred)))
print('Recall\t\t: {:.2}'.format(recall_score(y_test, y_pred)))
print('Accuracy\t: {:.2}'.format(accuracy_score(y_test, y_pred)))
print('F1-Score\t: {:.2}'.format(f1_score(y_test, y_pred)))

print('\n')

# Statistik Hasil Percobaan menggunakan ngram
# print('Statistik Hasil Data Dan Fitur Menggunakan ngram : ')
# print(f'Sel. Fitur\t: {fs_label}')
# print(f'Param. SVM\t: Kernel={pKernel[ik]}, C={pC[ic]}, Gamma={pGamma[ig]}')
# Statistik Hasil Data Dan Fitur Menggunakan ngram
# print(f'Jml. Data\t: {ngram_train_matrix.shape[0]} (80%)')
# print(f'Jml. Fitur\t: {ngram_train_matrix.shape[1]}')
#
# print('Precision\t: {:.2}'.format(precision_score(y_test, y_pred)))
# print('Recall\t\t: {:.2}'.format(recall_score(y_test, y_pred)))
# print('Accuracy\t: {:.2}'.format(accuracy_score(y_test, y_pred)))
# print('F1-Score\t: {:.2}'.format(f1_score(y_test, y_pred)))

# Simpan Model
filename = f'model-svm-{fs_label}-{pKernel[ik]}-{pC[ic]}.pickle'
pickle.dump(svm_classifier, open(filename, 'wb'))

vectorizer = tfidf_vectorizer
vectorizer.stop_words_ = None
clf = svm_classifier

with open(filename, 'wb') as fout:
    if pFitur[fs] == 'chisquare':
        pickle.dump((vectorizer, chi2, clf), fout)
    elif pFitur[fs] == 'randomforest':
        pickle.dump((vectorizer, chi2, clf), fout)
    else :
        pickle.dump((vectorizer, clf), fout)

print(f'Nama Model\t: {filename}')
