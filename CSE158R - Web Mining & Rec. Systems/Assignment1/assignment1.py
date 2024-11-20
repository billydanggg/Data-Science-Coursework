#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gzip
import numpy as np
import pandas as pd
from collections import defaultdict

def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)

def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        yield l.strip().split(',')


# ### Rating Prediction Task
# ___

# In[9]:


def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        yield l.strip().split(',')

allRatings = []
ratingsTrain = []
userRatings = defaultdict(list)
bookRatings = defaultdict(list)

for user, book, r in readCSV("train_Interactions.csv.gz"):
    r = int(r)
    allRatings.append(r)
    ratingsTrain.append((user, book, r))
    userRatings[user].append(r)
    bookRatings[book].append(r)

globalAverage = sum(allRatings) / len(allRatings)

best_lambda = 0.1  
num_epochs = 50
learning_rate = 0.005
latent_dim = 12 

alpha = globalAverage

beta_u = defaultdict(float)
beta_i = defaultdict(float)

user_factors = defaultdict(lambda: [0.1] * latent_dim)
book_factors = defaultdict(lambda: [0.1] * latent_dim)

for epoch in range(num_epochs):
    for user, book, rating in ratingsTrain:
        #dot product of latent factors
        dot_product = sum([pu * qi for pu, qi in zip(user_factors[user], book_factors[book])])

        prediction = alpha + beta_u[user] + beta_i[book] + dot_product
        error = rating - prediction

        alpha += learning_rate * error

        beta_u[user] += learning_rate * (error - best_lambda * beta_u[user])
        beta_i[book] += learning_rate * (error - best_lambda * beta_i[book])


        for k in range(latent_dim):

            user_factor_k = user_factors[user][k]
            book_factor_k = book_factors[book][k]

            user_factors[user][k] += learning_rate * (error * book_factor_k - best_lambda * user_factor_k)
            book_factors[book][k] += learning_rate * (error * user_factor_k - best_lambda * book_factor_k)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs} completed.")

predictions = open("predictions_Rating.csv", 'w')
for l in open("pairs_Rating.csv"):
    if l.startswith("userID"):
        predictions.write(l)  # Write header
        continue
    user, book = l.strip().split(',')

    dot_product = sum([pu * qi for pu, qi in zip(user_factors[user], book_factors[book])])

    prediction = alpha + beta_u.get(user, 0) + beta_i.get(book, 0) + dot_product

    prediction = max(1, min(5, prediction))

    predictions.write(f"{user},{book},{prediction}\n")

predictions.close()

print("Predictions saved to 'predictions_Rating.csv'.")


# ### Read Prediction Task
# ____

# In[3]:


from sklearn.model_selection import train_test_split

def readCSV(path):
    with gzip.open(path, 'rt') if path.endswith('.gz') else open(path, 'r') as f:
        for line in f:
            yield line.strip().split(',')

def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0

bookToUsers = defaultdict(set)
userToBooks = defaultdict(set)
bookCount = defaultdict(int)

for user, book, _ in readCSV("train_Interactions.csv.gz"):
    bookToUsers[book].add(user)
    userToBooks[user].add(book)
    bookCount[book] += 1


mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort(reverse=True)

top_blank_percent = int(len(mostPopular) * 0.30)
return1 = set(book for _, book in mostPopular[:top_blank_percent])

threshold = 0.005  
predictions = open("predictions_Read.csv", 'w')

for l in open("pairs_Read.csv"):
    if l.startswith("userID"):
        predictions.write(l)  
        continue
    u, b = l.strip().split(',')
    
    popular_prediction = b in return1

    jaccard_prediction = False
    if u in userToBooks:
        max_similarity = 0
        for b_prime in userToBooks[u]:
            max_similarity = max(max_similarity, jaccard_similarity(bookToUsers[b], bookToUsers[b_prime]))
        jaccard_prediction = max_similarity > threshold

#hybrid prediction 
    final_prediction = popular_prediction or jaccard_prediction


    predictions.write(f"{u},{b},{int(final_prediction)}\n")

predictions.close()

