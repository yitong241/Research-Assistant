import os
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer

porter = PorterStemmer()


fd = open("keywords.txt")
with open("processed.txt",'a+', encoding='utf-8') as f:
    data = fd.readlines()
    for line in data:
        L = []
        words = nltk.word_tokenize(line)
        for word in words:
            L.append(porter.stem(word))
        f.write(" ".join(L))
        f.write("\n")
fd.close()
f.close()

