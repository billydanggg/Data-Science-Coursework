#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy
import string
import random
import string
from sklearn import linear_model


# In[3]:


import warnings
warnings.filterwarnings("ignore")


# In[5]:


def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[7]:


def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)


# In[18]:


def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r


# In[20]:


answers = {}


# In[22]:


# Some data structures that will be useful


# In[26]:


allRatings = []
for l in readCSV("data/train_Interactions.csv.gz"):
    allRatings.append(l)


# In[28]:


len(allRatings)


# In[30]:


ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))


# In[32]:


##################################################
# Read prediction                                #
##################################################


# In[36]:


# Copied from baseline code -> recommendations made based on most popular items 
bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in readCSV("data/train_Interactions.csv.gz"):
    bookCount[book] += 1
    totalRead += 1

mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > totalRead/2: break


# In[44]:


#Personal EDA
allRatings[0]


# In[ ]:


### Question 1


# In[50]:


#create a set of all unique books from the training data
allBooks = set()
for _, book, _ in ratingsTrain:
    allBooks.add(book)

#for each user in the validation set, create a (user, book) pair with a book that the user has not interacted w/
negativeSamples = []
for u, b, _ in ratingsValid:
    #get the set of books the user has already interacted with
    readBooks = set(b for b, _ in ratingsPerUser[u])
    #find books the user hasn't read
    unreadBooks = list(allBooks - readBooks)
    if unreadBooks:
        #randomly select a book the user hasn't read as a negative sample
        negativeBook = random.choice(unreadBooks)
        negativeSamples.append((u, negativeBook))


# In[56]:


#combine the pos & neg samples and evaluate the accuracy of the baseline model below
correctPredictions = 0
totalPredictions = 0

#check positive samples (true interactions)
for u, b, _ in ratingsValid:
    if b in return1:  # If the book is in the set of popular books
        correctPredictions += 1
    totalPredictions += 1

#check negative samples (non-interactions)
for u, b in negativeSamples:
    if b not in return1:  # If the book is not in the set of popular books
        correctPredictions += 1
    totalPredictions += 1

#calculate accuracy
accuracy = correctPredictions / totalPredictions
print("Accuracy of the baseline model:", accuracy)
acc1 = accuracy #.7146


# In[58]:


answers['Q1'] = acc1


# In[60]:


assertFloat(answers['Q1'])


# In[ ]:


### Question 2


# In[68]:


#initialize variables to track the best threshold and its accuracy
bestThreshold = None
bestAccuracy = 0

#total number of interactions to adjust the threshold (e.g., vary from 10% to 90%)
thresholdPercentages = [0.1 * i for i in range(1, 10)]

for percentage in thresholdPercentages:
    #calculate the threshold based on the percentage of total interactions
    currentThreshold = totalRead * percentage

    #create the 'popular' book set based on the current threshold
    returnSet = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        returnSet.add(i)
        if count > currentThreshold:
            break

    #evaluate the accuracy of the model with the current threshold
    correctPredictions = 0
    totalPredictions = 0

    #check positive samples (true interactions)
    for u, b, _ in ratingsValid:
        if b in returnSet:  # If the book is in the set of popular books
            correctPredictions += 1
        totalPredictions += 1

    #check negative samples (non-interactions)
    for u, b in negativeSamples:
        if b not in returnSet:  # If the book is not in the set of popular books
            correctPredictions += 1
        totalPredictions += 1

    #calculate accuracy
    accuracy = correctPredictions / totalPredictions

    #update the best threshold if the current accuracy is higher
    if accuracy > bestAccuracy:
        bestAccuracy = accuracy
        bestThreshold = currentThreshold

print("Best threshold (in terms of totalRead):", bestThreshold / totalRead)
print("Best accuracy of the model with improved threshold:", bestAccuracy)
threshold = (bestThreshold / totalRead)
acc2 = bestAccuracy


# In[70]:


answers['Q2'] = [threshold, acc2] #[.7, .7569]


# In[72]:


assertFloat(answers['Q2'][0])
assertFloat(answers['Q2'][1])


# In[ ]:


### Question 3/4


# In[108]:


#evaluate the model with different Jaccard similarity thresholds
bestThreshold = None
bestAccuracy = 0
thresholds = [0.001 * i for i in range(1, 15)]  #consider thresholds from 0.001 to 0.015

for threshold in thresholds:
    correctPredictions = 0
    totalPredictions = 0

    #check positive samples (true interactions)
    for u, b, _ in ratingsValid:
        if u not in ratingsPerUser:
            #if user is not in the training data, max Jaccard should be 0 (improved model alot)
            maxSimilarity = 0
        else:
            #get the books this user has read in the training data
            userBooks = [book for book, _ in ratingsPerUser[u]]
            
            #compute the maximum Jaccard similarity for this (u, b) pair
            maxSimilarity = max((jaccard_similarity(b, b_prime) for b_prime in userBooks), default=0)

        #predict 'read' if the maximum similarity exceeds the threshold
        if maxSimilarity > threshold:
            correctPredictions += 1
        totalPredictions += 1

    #check negative samples (non-interactions)
    for u, b in negativeSamples:
        if u not in ratingsPerUser:
            #if user is not in the training data, max Jaccard should be 0
            maxSimilarity = 0
        else:
            #get the books this user has read in the training data
            userBooks = [book for book, _ in ratingsPerUser[u]]
            
            #compute the maximum Jaccard similarity for this (u, b) pair
            maxSimilarity = max((jaccard_similarity(b, b_prime) for b_prime in userBooks), default=0)

        # Predict 'not read' if the maximum similarity does not exceed the threshold
        if maxSimilarity <= threshold:
            correctPredictions += 1
        totalPredictions += 1

    accuracy = correctPredictions / totalPredictions

    #update the best threshold if the current accuracy is higher
    if accuracy > bestAccuracy:
        bestAccuracy = accuracy
        bestThreshold = threshold

# Report the best threshold and its performance
print("Best Jaccard similarity threshold:", bestThreshold)
print("Accuracy of the model with the best Jaccard similarity threshold:", bestAccuracy)


# In[76]:


#initialize variables to track the best combined threshold and its accuracy
bestCombinedThresholds = None
bestCombinedAccuracy = 0

#create the 'popular' book set for a given percentage threshold (e.g., 50% of total interactions)
def create_popular_set(percentage):
    threshold = totalRead * percentage
    popularSet = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        popularSet.add(i)
        if count > threshold:
            break
    return popularSet

#define possible threshold values for popularity and Jaccard similarity
popularityThresholds = [0.1 * i for i in range(1, 10)]
jaccardThresholds = [0.1 * i for i in range(1, 10)]

for popThreshold in popularityThresholds:
    popularSet = create_popular_set(popThreshold)

    for jaccardThreshold in jaccardThresholds:
        correctPredictions = 0
        totalPredictions = 0

        #check positive samples (true interactions)
        for u, b, _ in ratingsValid:
            if u not in ratingsPerUser:
                continue  # Skip users who do not appear in the training data

            #get the books this user has read in the training data
            userBooks = [book for book, _ in ratingsPerUser[u]]

            #compute the maximum Jaccard similarity for this (u, b) pair
            maxSimilarity = max((jaccard_similarity(b, b_prime) for b_prime in userBooks), default=0)

            #predict 'read' if the book is in the popular set or if the max Jaccard similarity exceeds its threshold
            if b in popularSet or maxSimilarity > jaccardThreshold:
                correctPredictions += 1
            totalPredictions += 1

        #evaluate on negative samples
        for u, b in negativeSamples:
            if u not in ratingsPerUser:
                continue  # Skip users who do not appear in the training data

            #get the books this user has read in the training data
            userBooks = [book for book, _ in ratingsPerUser[u]]

            #compute the maximum Jaccard similarity for this (u, b) pair
            maxSimilarity = max((jaccard_similarity(b, b_prime) for b_prime in userBooks), default=0)

            #predict 'not read' if the book is not in the popular set and the max Jaccard similarity does not exceed its threshold
            if b not in popularSet and maxSimilarity <= jaccardThreshold:
                correctPredictions += 1
            totalPredictions += 1

        accuracy = correctPredictions / totalPredictions

        #update the best thresholds if the current accuracy is higher
        if accuracy > bestCombinedAccuracy:
            bestCombinedAccuracy = accuracy
            bestCombinedThresholds = (popThreshold, jaccardThreshold)

print("Best popularity threshold (in terms of totalRead):", bestCombinedThresholds[0])
print("Best Jaccard similarity threshold:", bestCombinedThresholds[1])
print("Best accuracy of the combined model:", bestCombinedAccuracy)


# In[110]:


acc3 = bestAccuracy
acc4 = bestCombinedAccuracy


# In[112]:


answers['Q3'] = acc3
answers['Q4'] = acc4 #.7571


# In[96]:


assertFloat(answers['Q3'])
assertFloat(answers['Q4'])


# In[100]:


predictions = open("data/predictions_Read.csv", 'w')
for l in open("data/pairs_Read.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    # (etc.)

predictions.close()


# In[102]:


answers['Q5'] = "I confirm that I have uploaded an assignment submission to gradescope"


# In[104]:


assert type(answers['Q5']) == str


# In[ ]:


##################################################
# Rating prediction                              #
##################################################


# In[ ]:





# In[ ]:


### Question 6


# In[116]:


# Step 1: Split the training data into training and validation sets
ratingsTrain = allRatings[:190000]  # First 190,000 for training
ratingsValid = allRatings[190000:]  # Last 10,000 for validation

# Step 2: Initialize global mean (alpha), user biases (beta_u), and item biases (beta_i)
alpha = np.mean([r for _, _, r in ratingsTrain])  # Global mean rating
beta_u = defaultdict(float)  # User biases initialized to 0
beta_i = defaultdict(float)  # Item biases initialized to 0

# Regularization parameter
lambda_reg = 1

# Step 3: Optimize using gradient descent or coordinate descent
num_epochs = 50
learning_rate = 0.005

for epoch in range(num_epochs):
    for u, i, r in ratingsTrain:
        # Compute prediction and error
        prediction = alpha + beta_u[u] + beta_i[i]
        error = r - prediction
        
        # Update biases using gradient descent
        beta_u[u] += learning_rate * (error - lambda_reg * beta_u[u])
        beta_i[i] += learning_rate * (error - lambda_reg * beta_i[i])
    
    # Optional: Print progress every few epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs} completed.")

# Step 4: Calculate MSE on the validation set
squared_errors = []
for u, i, r in ratingsValid:
    prediction = alpha + beta_u[u] + beta_i[i]
    squared_errors.append((r - prediction) ** 2)

mse = np.mean(squared_errors)
print("Mean Squared Error on the validation set:", mse)
validMSE = mse


# In[117]:


answers['Q6'] = validMSE


# In[118]:


assertFloat(answers['Q6'])


# In[ ]:


### Question 7


# In[136]:


#find the user with the largest beta_u value
maxUser = str(max(beta_u, key=beta_u.get))
maxBeta = float(beta_u[maxUser])

#find the user with the smallest (most negative) beta_u value
minUser = str(min(beta_u, key=beta_u.get))
minBeta = float(beta_u[minUser])

print("User with the largest beta_u:")
print(f"User ID: {maxUser}, Beta value: {maxBeta}")

print("\nUser with the smallest (most negative) beta_u:")
print(f"User ID: {minUser}, Beta value: {minBeta}")


# In[138]:


answers['Q7'] = [maxUser, minUser, maxBeta, minBeta]


# In[140]:


assert [type(x) for x in answers['Q7']] == [str, str, float, float]


# In[ ]:


### Question 8


# In[144]:


#define a function to train the model with a given lambda and calculate the MSE
def train_model(lambda_reg, num_epochs=50, learning_rate=0.005):
    alpha = np.mean([r for _, _, r in ratingsTrain])  # Global mean rating
    beta_u = defaultdict(float)  # User biases initialized to 0
    beta_i = defaultdict(float)  # Item biases initialized to 0

    #training loop
    for epoch in range(num_epochs):
        for u, i, r in ratingsTrain:
            # Compute prediction and error
            prediction = alpha + beta_u[u] + beta_i[i]
            error = r - prediction

            # Update biases using gradient descent with regularization
            beta_u[u] += learning_rate * (error - lambda_reg * beta_u[u])
            beta_i[i] += learning_rate * (error - lambda_reg * beta_i[i])

    # Calculate MSE on the validation set
    squared_errors = []
    for u, i, r in ratingsValid:
        prediction = alpha + beta_u[u] + beta_i[i]
        squared_errors.append((r - prediction) ** 2)

    mse = np.mean(squared_errors)
    return mse, alpha, beta_u, beta_i

# Step 3: Try different values of lambda and find the best one
lambda_values = [0.1, 0.5, 1, 2, 5, 10]
best_lambda = None
best_mse = float('inf')
best_alpha = None
best_beta_u = None
best_beta_i = None

for lambda_reg in lambda_values:
    mse, alpha, beta_u, beta_i = train_model(lambda_reg)
    print(f"Lambda: {lambda_reg}, MSE: {mse}")
    if mse < best_mse:
        best_mse = mse
        best_lambda = lambda_reg
        best_alpha = alpha
        best_beta_u = beta_u
        best_beta_i = beta_i

lamb = best_lambda
validMSE = best_mse
print("\nBest Lambda:", best_lambda)
print("Valid MSE on the validation set with the best lambda:", best_mse)


# In[145]:


answers['Q8'] = (lamb, validMSE)


# In[146]:


assertFloat(answers['Q8'][0])
assertFloat(answers['Q8'][1])


# In[158]:


predictions = open("data/predictions_Rating.csv", 'w')
for l in open("data/pairs_Rating.csv"):
    if l.startswith("userID"): # header
        predictions.write(l)
        continue
    u, b = l.strip().split(',')
    
    # Make the prediction using the best alpha, beta_u, and beta_i values
    prediction = best_alpha + best_beta_u[u] + best_beta_i[b]
    
    # Clip the prediction to a valid rating range if necessary (e.g., between 1 and 5)
    prediction = max(1, min(5, prediction))
    
    # Write the user, book, and predicted rating to the output file
    predictions.write(f"{u},{b},{prediction}\n")
    
predictions.close()


# In[160]:


f = open("answers_hw3.txt", 'w')
f.write(str(answers) + '\n')
f.close()

