import nltk
from nltk.corpus import stopwords
import numpy as np
import re
import keras
from os.path import join
from jpype import JClass, JString, getDefaultJVMPath, shutdownJVM, startJVM, java
from typing import List
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import  accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import tensorflow
from numpy import array
from keras.utils import to_categorical
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Activation, Bidirectional
from keras.layers import LSTM
from keras.layers import Embedding, MaxPooling1D
from keras.layers import Flatten, AveragePooling1D
from keras.layers import Dropout, TimeDistributed
from keras.models import Model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from keras import backend

import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"


with open(r'headers.txt') as f:
    headers = f.readlines()

with open(r'labels.txt') as f:
    label_list = f.readlines()
    
    
for i in range(len(headers)):

    headers[i] = headers[i].lower()
    headers[i] = headers[i].split()
    
nltk.download("stopwords")
with open(r'headers.txt') as f:
    headers = f.readlines()

with open(r'labels.txt') as f:
    label_list = f.readlines()
    
    
for i in range(len(headers)):

    headers[i] = headers[i].lower()
    headers[i] = headers[i].split()
    
nltk.download("stopwords")

def removeStopWords(headers):

    filtered_stopwords = []
    filtered_stopwords_number = []

    stop_words = stopwords.words('turkish')

    stop_words.append("bir")
    stop_words.append("iki")
    stop_words.append("üç")
    stop_words.append("dört")
    stop_words.append("beş")
    stop_words.append("altı")
    stop_words.append("yedi")
    stop_words.append("sekiz")
    stop_words.append("dokuz")
    stop_words.append("on")
    stop_words.append("ancak")
    stop_words.append("artık")
    stop_words.append("asla")
    stop_words.append("bana")
    stop_words.append("bazen")
    stop_words.append("bazıları")
    stop_words.append("bazısı")
    stop_words.append("ben")
    stop_words.append("beni")
    stop_words.append("benim")
    stop_words.append("bile")
    stop_words.append("böyle")
    stop_words.append("böylece")
    stop_words.append("bütün")
    stop_words.append("burada")
    stop_words.append("bunun")
    stop_words.append("bunu")
    stop_words.append("çoğu")
    stop_words.append("çoğuna")
    stop_words.append("çoğunu")
    stop_words.append("değil")
    stop_words.append("demek")
    stop_words.append("diğer")
    stop_words.append("dolayı")
    stop_words.append("elbette")
    stop_words.append("madem")
    stop_words.append("nesi")
    stop_words.append("zaten")
    stop_words.append("zira")
    stop_words.append("yoksa")
    stop_words.append("yine")
    stop_words.append("yerine")
    stop_words.append("veyahut")
    stop_words.append("var")
    stop_words.append("üzere")
    stop_words.append("tamam")
    stop_words.append("tümü")
    stop_words.append("tabi")
    stop_words.append("rağmen")
    stop_words.append("oysa")
    stop_words.append("oysaki")
    stop_words.append("orada")
    stop_words.append("öbürü")
    stop_words.append("önce")
    stop_words.append("içinde")
    stop_words.append("işte")
    stop_words.append("gene")
    stop_words.append("falan")
    stop_words.append("felan")
    stop_words.append("filan")
    stop_words.append("fakat")
    stop_words.append("hala")
    stop_words.append("hangi")
    stop_words.append("hangisi")
    stop_words.append("hani")
    stop_words.append("hatta")
    stop_words.append("henüz")
    stop_words.append("hepsine")
    stop_words.append("hepsini")
    stop_words.append("herkes")
    stop_words.append("hiçbiri")




    print("stop_words : ",stop_words)



    for i in headers:
        filtered_sentence = [w for w in i if not w in stop_words]

        filtered_stopwords.append(" ".join(filtered_sentence))

    return filtered_stopwords,filtered_stopwords_number
    
    
    
    
headers,filtered_stopwords_number = removeStopWords(headers)


for i in range(len(headers)):

    headers[i] = re.sub(r'[0-9]'," ",headers[i])
    headers[i] = re.sub(r'[.,x?!<=>&*%+^“/”):-;‘’"’(]'," ", headers[i])
    
    
ZEMBEREK_PATH = 'zemberek-full.jar'
startJVM(getDefaultJVMPath(), '-ea', '-Djava.class.path=%s' % (ZEMBEREK_PATH))

TurkishMorphology = JClass('zemberek.morphology.TurkishMorphology')
Paths = JClass('java.nio.file.Paths')
morphology = TurkishMorphology.createWithDefaults()

TurkishMorphology: JClass = JClass('zemberek.morphology.TurkishMorphology')

morphology: TurkishMorphology = TurkishMorphology.createWithDefaults()

for i in range(len(headers)):

    sentence: str = headers[i]

    analysis: java.util.ArrayList = (
        morphology.analyzeAndDisambiguate(sentence).bestAnalysis()
        )

    pos: List[str] = []

    for j, analysis in enumerate(analysis, start=1):
        #print(
        f'\nAnalysis {j}: {analysis}',
        f'\nPrimary POS {j}: {analysis.getPos()}'
        f'\nPrimary POS (Short Form) {j}: {analysis.getPos().shortForm}'
            #)
        pos.append(
        f'{str(analysis.getLemmas()[0])}'
            #f'-{analysis.getPos().shortForm}'
                )
    f'\nFull sentence with POS tags: {" ".join(pos)}'
    headers[i] = pos
    
max_len = 1000
tok = Tokenizer(num_words=max_len)
tok.fit_on_texts(headers)
sequences = tok.texts_to_sequences(headers)
headers = sequence.pad_sequences(sequences,maxlen=max_len)

le = LabelEncoder()
label_list = le.fit_transform(label_list)
label_list = label_list.reshape(-1,1)

seed = 7
np.random.seed(seed)

precision_value_train = []
precision_value_test = []
recall_value_train = []
recall_value_test = []
conf_matrix_list_of_arrays = []

maxlen = 1000
vocab_size= 100000

skf = StratifiedKFold(n_splits=10)

accSum = 0
val_accSum = 0
sayac=0
accList = []
accListVal = []

for train, test in skf.split(headers, label_list):

    sayac = sayac+1
    embedding_dim=300


    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
    model.add(layers.Conv1D(64, 3, activation='relu'))
    model.add(layers.MaxPooling1D())
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam' ,metrics=['accuracy'])
    history = model.fit(headers[train],label_list[train], validation_data = (headers[test], label_list[test]), batch_size=128,epochs=2)

    pred = model.predict(headers[test])
    pred_train = model.predict(headers[train])

    precision_train = precision_score(label_list[train], pred_train.round() , average='macro')
    precision_value_train.append(precision_train)
    precision_test = precision_score(label_list[test],pred.round() , average='macro')
    precision_value_test.append(precision_test)


    recall_train = precision_score(label_list[train], pred_train.round() , average='macro')
    recall_value_train.append(recall_train)
    recall_test = recall_score(label_list[test], pred.round(), average='macro')
    recall_value_test.append(recall_test)


    conf_matrix = confusion_matrix(label_list[test].argmax(axis=1), pred.round().argmax(axis=1))
    conf_matrix_list_of_arrays .append(conf_matrix)

    history.history.keys()
    print("\n")
    epochDizisi = history.history['accuracy']
    accList.append(history.history['accuracy'])
    accListVal.append(history.history['val_accuracy'])
    valDizisi = history.history['val_accuracy']
    accSum += epochDizisi[len(epochDizisi)-1]
    val_accSum += valDizisi[len(valDizisi)-1]
    print("\n")




print("\n Total Sayac: ", sayac)
accMean=accSum/sayac
val_accMean= val_accSum/sayac
print("\n Total Train Accuracy:", accMean)
print("\n Total Test Val_accuracy:", val_accMean)

print("Precision Train: ",np.mean(precision_value_train))
print("Precision Test: ",np.mean(precision_value_test))
print("Recall Train: ",np.mean(recall_value_train))
print("Recall Test: ",np.mean(recall_value_test))

confision_matrix = np.mean(conf_matrix_list_of_arrays, axis=0)


print("cnn Train std Accuracy: ", np.std(accList))
print("cnn Test std Accuracy: ",np.std(accListVal))
print("cnn Train std Precision: ",np.std(precision_value_train))
print("cnn Test std Precision: ",np.std(precision_value_test))
print("cnn Train std Recall : ",np.std(recall_value_train))
print("cnn Test std Recall : ",np.std(recall_value_test))


model.summary()