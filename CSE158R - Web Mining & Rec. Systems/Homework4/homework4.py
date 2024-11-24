#!/usr/bin/env python
# coding: utf-8

# In[153]:


import gzip
import math
import numpy
import random
import sklearn
import string
import numpy as np
from collections import defaultdict
from nltk.stem.porter import *
from sklearn import linear_model
from gensim.models import Word2Vec
import dateutil
from scipy.sparse import lil_matrix # To build sparse feature matrices, if you like


# In[4]:


answers = {}


# In[6]:


def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# ### Question 1
# Using the Steam category data, build training/test sets consisting of 10,000 reviews each. Code to do so is provided in the stub. We'll start by building features to represent the common words. Start by removing punctuation and capitalization, and finding the 1,000 most common words across all reviews ('text' field) in the training set. See the 'text mining' lectures for code for this process. Report the 10 most common words, along with their frequencies, as a list of (frequncy, word) tuples.

# In[10]:


dataset = []

f = gzip.open("steam_category.json.gz")
for l in f:
    d = eval(l)
    dataset.append(d)
    if len(dataset) >= 20000:
        break
        
f.close()


# In[43]:


#Split data in training and test set
Ntrain = 10000
Ntest = 10000

dataTrain = dataset[:Ntrain]
dataTest = dataset[Ntrain:Ntrain + Ntest]


# In[39]:


dataTrain


# In[49]:


#initialize dict of word:appearance_count
wordCount = defaultdict(int)

#set of punctuation to reference
sp = set(string.punctuation)

for d in dataTrain: #for review in dataset
    r = ''.join([c for c in d['text'].lower() if not c in sp]) #forms a list of all words (remove capitalization) that are not punctuation
    ws = r.split() #splits that list into words
    for w in ws: #traverses through words & adds a count to dict. for each appearance
        wordCount[w]+=1


# In[67]:


#create list of dict. entries as can't .sort() dict. type -> sort from most to least common
counts = [(wordCount[w], w) for w in wordCount]
counts.sort(reverse = True)

#grab list of words where x is in top 100 spots of sorted list, this will be a b.o.w feature vector for models
common_words = [x[1] for x in counts[:1000]]


# In[69]:


answers['Q1'] = counts[:10]


# In[71]:


assertFloatList([x[0] for x in answers['Q1']], 10)


# ### Question 2
# Build bag-of-words feature vectors by counting the instances of these 1,000 words in each review. Set the labels (y) to be the 'genreID' column for the training instances. You may use these lab|els directly with sklearn's LogisticRegression model, which will automatically perform multiclass classification. Report performance (accuracy) on your test set.

# In[73]:


NW = 1000 # dictionary size


# In[95]:


wordId = dict(zip(common_words, range(len(words))))
wordSet = set(common_words)


# In[181]:


#building X
def feature(datum): #function that creates b.o.w feature
    feat = [0]*len(words) #initialize feature vector of length 1000, all currently 0 count
    review = ''.join([c for c in datum['text'].lower() if not c in sp])
    for w in review.split():
        if w in wordSet:
            feat[wordId[w]] += 1
        feat.append(0) #offset during loop
        return feat

X = [feature(d) for d in dataset] #feature vectors for entire dataset


# In[183]:


y = [d['genre'] for d in dataset] #response variable for entire dataset


# In[197]:


#subset for train/test split
Xtrain = X[:Ntrain]
ytrain = y[:Ntrain]
Xtest = X[Ntrain:]
ytest = y[Ntrain:]


# In[201]:


print([x for x in Xtrain if x is None])


# In[193]:


mod = linear_model.LogisticRegression(C=1)
mod.fit(Xtrain, ytrain)


# In[ ]:





# In[ ]:


answers['Q2'] = sum(correct) / len(correct)


# In[ ]:


assertFloat(answers['Q2'])


# ### Question 3
# What is the inverse document frequency of the words 'character', 'game', 'length', 'a', and 'it'? What are their td-idf scores in the first (training) review (using log base 10, unigrams only, following the first definition of tf-idf given in the slides)? All frequencies etc. should be calculated using the training data only. Your answer should be a list of five (idf, tfidf) pairs.

# In[ ]:





# In[ ]:


answers['Q3'] = 


# In[ ]:


assertFloatList([x[0] for x in answers['Q3']], 5)
assertFloatList([x[1] for x in answers['Q3']], 5)


# ### Question 4
# Adapt your unigram model to use the tdidf scores of words, rather than a bag-of-words representation. That is, rather than your features containing the word counts for the 1000 most common unigrams. Report the accuracy of this new model.

# In[ ]:


# Build X and y...


# In[ ]:


Xtrain = X[:Ntrain]
ytrain = y[:Ntrain]
Xtest = X[Ntrain:]
ytest = y[Ntrain:]


# In[ ]:


mod = linear_model.LogisticRegression(C=1)


# In[ ]:





# In[ ]:


answers['Q4'] = sum(correct) / len(correct)


# In[ ]:


assertFloat(answers['Q4'])


# ### Question 5
# Which review in the test set has the highest cosine similarity compared to the first review in the training set, in terms of their tf-idf representation (considering unigrams only). Provide the cosine similarity score and the reviewID.

# In[ ]:


def Cosine(x1,x2):
    # ...


# In[ ]:





# In[ ]:


similarities.sort(reverse=True)


# In[ ]:


answers['Q5'] = similarities[0]


# In[ ]:


assertFloat(answers['Q5'][0])


# ### Question 6
# Try to improve upon the performance of the above classifiers from questions 2 and 4 by using different dictionary sizes, or changing the regularization constant C passed to the logistic regression model. Report the performance of your solution.
# 
# Use the first half (10,000) of the corpus for training and the rest for testing (code to read the data is provided in the stub). Process review without capitalization or punctuation (and without using stemming or stopwords).

# In[ ]:





# In[ ]:


answers['Q6'] = 


# In[ ]:


assertFloat(answers['Q6'])


# ### Question 7
# This task should be completed using the entire dataset of 20,000 reviews from Goodreads:
# 
# Using the word2vec library in gensim, fit an item2vec model, treating each sentence as a temporally-ordered list of items per user. Use parameters min_count = 1, size = 5, window = 3, sg = 1. Report the 5 most similar items to the book from the first review along with their similarity scores (your answer can be the output of the similar_by_word function).

# In[ ]:


import dateutil.parser


# In[ ]:


dataset = []

f = gzip.open("young_adult_20000.json.gz")
for l in f:
    d = eval(l)
    d['datetime'] = dateutil.parser.parse(d['date_added'])
    dataset.append(d)
    if len(dataset) >= 20000:
        break
        
f.close()


# In[ ]:





# In[ ]:


model5 = Word2Vec(reviewLists,
                  min_count=1, # Words/items with fewer instances are discarded
                  vector_size=5, # Model dimensionality
                  window=3, # Window size
                  sg=1) # Skip-gram model


# In[ ]:





# In[ ]:


answers['Q7'] = res[:5]


# In[ ]:


assertFloatList([x[1] for x in answers['Q7']], 5)


# In[ ]:


f = open("answers_hw4.txt", 'w')
f.write(str(answers) + '\n')
f.close()


# In[ ]:




