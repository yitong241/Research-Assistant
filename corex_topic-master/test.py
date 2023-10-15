import json
import csv

import pandas as pd
import numpy as np
import scipy.sparse as ss
import matplotlib.pyplot as plt

import corextopic.corextopic as ct
import corextopic.vis_topic as vt # jupyter notebooks will complain matplotlib is being loaded twice

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import entropy

# load the data from file
def load_tsv(filename):
    res = []
    data = pd.read_csv(filename, encoding='utf-8', sep="\t")
    votes = []
    actual_rating = []
    customer_id = []
    review_id = []
    for line in data.values:
        res.append(str(line[13]).lower())
        votes.append((int(line[8]), int(line[9] - int(line[8]))))
        actual_rating.append(int(line[7]))
        customer_id.append(line[1])
        review_id.append(line[2])
    return res, votes, actual_rating, customer_id, review_id


data, votes, actual_rating, customer_id, review_id = load_tsv("amazon_reviews_us_Digital_Video_Games_v1_00.tsv")

print(len(data))

# vectorize the words in the articles into binary
vectorizer = CountVectorizer(stop_words='english', max_features=20000, binary=True)
doc_word = vectorizer.fit_transform(data)
doc_word = ss.csr_matrix(doc_word)

num_topic = 10
# get all the words
words = list(np.asarray(vectorizer.get_feature_names_out()))

# get rid of all the digits
not_digit_inds = [ind for ind,word in enumerate(words) if not word.isdigit()]
doc_word = doc_word[:,not_digit_inds]
words    = [word for ind,word in enumerate(words) if not word.isdigit()]

# set the number of topics to be 10
topic_model = ct.Corex(n_hidden=num_topic, words=words, max_iter=2000, verbose=False, seed=1)
topic_model.fit(doc_word, words=words)

# Print all topics from the CorEx topic model
topics = topic_model.get_topics()
for n,topic in enumerate(topics):
    topic_words,_,_ = zip(*topic)
    print('{}: '.format(n) + ', '.join(topic_words))

prob = topic_model.predict_proba(doc_word)[0]


L = []
L.append(["customer_id", "review_id", "actual_rating", "upvote", "down_vote", "original_content", "prob_a1", "prob_a2", "prob_a3", "prob_a4", "prob_a5", "prob_a6", "prob_a7", "prob_a8", "prob_a9", "prob_a10", "entropy", "length"])
for i in range(144724):
    LL = []
    LL.append(customer_id[i])
    LL.append(review_id[i])
    LL.append(actual_rating[i])
    LL.append(votes[i][0])
    LL.append(votes[i][1])
    LL.append(data[i])
    for probaility in prob[i]:
        #LL.append("%.4f" % (probaility*100) + "%" + "  ")
        LL.append(probaility)
    ent = entropy(prob[i], base=2)
    LL.append(ent)
    LL.append(len(data[i].split()))
    

    L.append(LL)
    print(i)

with open('amazons.csv', 'a+', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerows(L)
    f.close()


