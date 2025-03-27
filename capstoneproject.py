#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 17:10:17 2023

@author: isabellawrobleski
"""

#%% initialization

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

n_number = 13967558

#%% data loader

pddata = pd.read_csv('spotify52kData.csv')
data = pddata.to_numpy()

'''
indices:
0 - song number
1 - artist(s)
2 - album name
3 - track name
4 - popularity
5 - duration
6 - explicit
7 - danceability
8 - energy
9 - key
    A- 0
    Bb - 1
    B - 2
    C - 3
    Db - 4
    D - 5
    Eb - 6
    E - 7
    F - 8
    F# - 9
    G - 10
    G# - 11
10 - loudness
11 - mode
    0 - minor
    1 - major
12 - speechiness
13 - acousticness
14 - instrumentalness
15 - liveness
16 - valence (upliftingness)
17 - tempo
18 - time signature
19 - genre
'''

# specific data columns:

popularity = data[:,4].astype(int)
duration = data[:,5].astype(int)
explicit = data[:,6].astype(int)
danceability = data[:,7].astype(float)
energy = data[:,8].astype(float)
key = data[:,9].astype(int)
loudness = data[:,10].astype(float)
mode = data[:,11].astype(int)
speechiness = data[:,12].astype(float)
acousticness = data[:,13].astype(float)
instrumentalness = data[:,14].astype(float)
liveness = data[:,15].astype(float)
valence = data[:,16].astype(float)
tempo = data[:,17].astype(float)
time_signature = data[:,18].astype(int)
genres = data[:,19]

# normalized columns:
    
zduration = stats.zscore(duration)
zdanceability = stats.zscore(danceability)
zenergy = stats.zscore(energy)
zloudness = stats.zscore(loudness)
zspeechiness = stats.zscore(speechiness)
zacousticness = stats.zscore(acousticness)
zinstrumentalness = stats.zscore(instrumentalness)
zliveness = stats.zscore(liveness)
zvalence = stats.zscore(valence)
ztempo = stats.zscore(tempo) 

predictors = [duration, danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo]
zpredictors = [zduration, zdanceability, zenergy, zloudness, zspeechiness, zacousticness, zinstrumentalness, zliveness, zvalence, ztempo]
predictornames = ["duration", "danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"]


#%% eda

print(stats.mode(data[:,1], axis=0, nan_policy='omit', keepdims=False).mode)
print('the average duration for these songs is', format(np.mean(data[:,5])/1000/60, '.2f'), 'minutes')
print(stats.mode(data[:,6], axis=0, nan_policy='omit', keepdims=False).mode)
print(stats.mode(key, axis=0, nan_policy='omit', keepdims=False).mode)
print('the average loudness for these songs is', np.mean(data[:,10]))
print(stats.mode(data[:,11], axis=0, nan_policy='omit', keepdims=False).mode)
print('the average tempo for these songs is', np.mean(data[:,17]))
print(stats.mode(data[:,18], axis=0, nan_policy='omit', keepdims=False).mode)

# the most common artist in this list is "my little airport"
# the average duration is almost 4 minutes
# most of these songs are non-explicit
# the most common key is E
# the average loudness is -7.972 dB
# most of the songs are in a major key
# the average tempo is 123.5, which is relatively fast
# the most common time signature is 4-based

genre_counts = pddata.value_counts(subset='track_genre')        # there are equal numbers of each genre

# i also think the choices of genre are interesting
# i wouldn't personally group together all artists of a certain nationality
# or tracks in a certain language, like french, british, etc.

speechiness_counts = pddata.value_counts(subset='speechiness')  # there are few tracks with high speechiness

# after examining the data further, most of the songs that i know in this dataset
# have a speechiness around .1 or less which was surprising

# examining popularity:
plt.hist(popularity, bins=100)
popularity_counts = pddata.value_counts(subset='popularity')
print(np.mean(popularity), np.median(popularity), np.std(popularity))
# a lot of songs have a popularity of 0
# few songs have a high popularity (above 80)


#%% question 1 - distributions

# we will plot a histogram for each of the ten predictors
# they have each been z-scored and the array 'zpredictors' defined above 
# contains all of the z-scored data

for ii in range(len(zpredictors)):
    plt.hist(zpredictors[ii], bins=80, color='darkcyan')
    nametemp = predictornames[ii].capitalize()
    plt.xlabel(f'{nametemp} Z-Score')
    plt.ylabel('Counts')
    plt.title(f'{nametemp} Distribution (Z-Scored)')
    plt.show()
    
# we will also plot an additional histogram for duration
# since there were extreme values in the original histogram
# so i would like to zoom in and look at the distribution of the bulk of the data 
    
plt.hist(zduration, bins=80, color='darkcyan', range=(-3.5,3.5))
plt.xlabel('Duration Z-Score')
plt.ylabel('Counts')
plt.title('Duration Distribution Center (Z-Scored)')
plt.show()

#%% question 2 - relationship between song length and popularity

# correlation is used to quantify relationships
# the relationship might not be linear, so we will use spearman's correlation

spearman_correlation = stats.spearmanr(duration, popularity)    # creating the object
rho = spearman_correlation.correlation                          # getting the correlation coefficient

# the spearman coefficient indicates that there is no strong monotonic relationship
# we will create a scatter plot to see if there is a relationship 

plt.scatter(duration, popularity, s=2, marker='o', color='rebeccapurple')
plt.xlabel('Duration (ms)')
plt.ylabel('Popularity Score')
plt.title(f'Relationship Between Track Duration and Popularity, rho = {rho:.3f}')
plt.show()

# there seems to be litle relationship since most of the tracks have a shorter duration
# however, there are no long tracks with high popularity
# and it looks as though after a certain duration there may be a negative relationship

#%% question 3 - explicitness and popularity

# null hypothesis: there is no difference in popularity between songs that are explicit and songs that are clean
# alternative hypothesis: songs that are explicit are more popular than songs that are clean

# from eda, i know that there is a large proportion of the data with popularity 0, so it is not reasonable to reduce to means
# these are also ratings, so i will be using a mann-whitney u test
# a one-tailed test will be used since the alternative hypothesis is that one group is more popular than the other

# creating subsets of the data for ecplicit tracks and non-explicit
explicit_tracks = data[np.where(data[:,6]==True)]
clean_tracks = data[np.where(data[:,6]==False)]

# isolating popularity columns:
explicit_popularities = explicit_tracks[:,4].astype(int)
clean_popularities = clean_tracks[:,4].astype(int)

# u-test:
u, p = stats.mannwhitneyu(explicit_popularities, clean_popularities, alternative='greater')

# histograms and effect size calculation:
plt.hist(explicit_popularities, bins=50, color='lightpink')
plt.axvline(np.median(explicit_popularities), color='maroon', label=f'Median Popularity: {np.median(explicit_popularities)}')
plt.xlabel('Popularity Score')
plt.ylabel('Count')
plt.title('Popularity Counts for Explicit Tracks')
plt.legend()
plt.show()

plt.hist(clean_popularities, bins=50, color='lightpink')
plt.axvline(np.median(clean_popularities), color='maroon', label=f'Median Popularity: {np.median(clean_popularities)}')
plt.xlabel('Popularity Score')
plt.ylabel('Count')
plt.title('Popularity Counts for Non-Explicit Tracks')
plt.legend()
plt.show()

d = (np.mean(explicit_popularities)-np.mean(clean_popularities))/np.std(popularity)
print(d)

# the p value is 1.53e-19 < 0.05, so there is statistical significance
# we can drop the assumption that the null hypothesis is true
# and explicit songs are more popular than non-explicit songs
# although the difference is quite small

#%% question 4 - mode and popularity

# null hypothesis: there is no difference in popularity between songs that are major and songs that are minor
# alternative hypothesis: songs that are in a major key are more popular than songs that are in a minor key

# from eda, i know that there is a large proportion of the data with popularity 0, so it is not reasonable to reduce to sample means
# these are ratings, so i will be using a mann-whitney u test

# a one-tailed test will be used
# since the alternative hypothesis is that one group is more popular than the other

# creating subsets of the data for each mode:
major_tracks = data[np.where(data[:,11]==True)]
minor_tracks = data[np.where(data[:,11]==False)]

# isolating the popularity column:
major_popularities = major_tracks[:,4].astype(int)
minor_popularities = minor_tracks[:,4].astype(int)

# running the u test:
u, p = stats.mannwhitneyu(major_popularities, minor_popularities, alternative='greater')

# the p value is 0.99, so there is not significance
# the null hypothesis cannot be dropped
# and nothing can be concluded (assuming the null hypothesis is true, this data is very plausible)

# since this is such a large p value, i would like to try it again with a new alternative hypothesis
# that songs in a minor key are more popular than songs in a major key

# running the u test:
u, p = stats.mannwhitneyu(minor_popularities, major_popularities, alternative='greater')

# in this case p = 0.000001 < 0.05 so there is significance
# and songs in a minor key are actually more popular than songs in a major key

# plotting popularities and computing effect size:
plt.hist(major_popularities, bins=50, color='lightpink')
plt.axvline(np.median(major_popularities), color='maroon', label=f'Median Popularity: {np.median(major_popularities)}')
plt.xlabel('Popularity Score')
plt.ylabel('Count')
plt.title('Popularity Counts for Major Tracks')
plt.legend()
plt.show()

plt.hist(minor_popularities, bins=50, color='lightpink')
plt.axvline(np.median(minor_popularities), color='maroon', label=f'Median Popularity: {np.median(minor_popularities)}')
plt.xlabel('Popularity Score')
plt.ylabel('Count')
plt.title('Popularity Counts for Minor Tracks')
plt.legend()
plt.show()

d = (np.mean(minor_popularities)-np.mean(major_popularities)) / np.std(popularity)
print(d)

#%% question 5 - relationship between energy and loudness

# we will be examining the relationship between loudness and energy
# to see if it would be plausible to predict energy based solely on loudness
# the relationship may not be linear, so we will first use spearman's correlation

spearman_correlation = stats.spearmanr(energy, loudness)    # creating the object
rho = spearman_correlation.correlation                      # getting the correlation coefficient

# the spearman correlation is 0.73, so there is a monotonic relationship
# we will now use pearson's correlation to see if this relationship is linear

pearson_correlation = np.corrcoef(energy, loudness)     # correlation matrix
r = pearson_correlation[0,1]                            # getting correlation coefficient
rsq = r**2

# the pearson correlation is 0.77, so there is a positive linear relationship

plt.scatter(loudness, energy, s=3, marker='o', color='cornflowerblue')
plt.xlabel('Loudness (average dB)')
plt.ylabel('Energy Score')
plt.title('Relationship Between Loudness and Track Energy')
plt.show()

# the relationship looks more quadratic than linear
# but there is definitely a relationship
# based on the pearson correlation, loudness accounts for about 60% of the variance in energy
# so a majority of the energy score is reflecting the loudness of a track

#%% question 6 - popularity predictions

# we want to know which predictor is the best when predicting popularity
# so we will compute r squared values for each predictor
# which tell us how much variability in popularity can be explained by the predictor 
    
# creating empty list for r-squared values
rsqs = []

for ii in range(len(predictors)):                       # iterating through all predictors
    x = predictors[ii].reshape(len(predictors[ii]), 1)  # set x as predictor
    y = popularity                                      # set y as popularity
    xtrain, xtest, ytrain, ytest = train_test_split(x ,y, test_size=0.2, random_state=n_number)
    model = LinearRegression().fit(xtrain,ytrain)       # fit linear regression model
    rsq = model.score(xtest,ytest)                      # get r-squared
    rsqs.append(rsq)                                    # add to list
    
max_rsq = max(rsqs)   # getting the maximum r-squared out of all 10 (0.019)
print(max_rsq)
rsqs.index(max(rsqs)) # getting the index so we know which predictor it is (6)

# this r-squared value corresponds to instrumentalness
# so instrumentalness is the best predictor for popularity out of these predictors
# even though it still only accounts for about 2% of the variability

# instrumentalness and popularity scatter plot: 
plt.scatter(instrumentalness, popularity, s=3, marker='o', color='goldenrod')
plt.xlabel('Instrumentalness Score')
plt.ylabel('Popularity Score')
plt.title('Track Instrumentalness vs Popularity')

#%% question 7 - multiple regresssion for popularity

# we want to use all of the predictors to predict popularity
# in order to avoid overfitting the model, we will use cross-validation
# we will also do a ridge regression

# first we will split the data
# we will use an 80/20 split since 20% of the data is still about 10,000 datapoints
x = np.transpose(predictors)
y = popularity
xtrain, xtest, ytrain, ytest = train_test_split(x ,y, test_size=0.2, random_state=n_number)

# next we will determine the optimal lambda for the ridge regression
ridge = Ridge()     # instantiation
rmses = []          # container for rmses
lams = range(500)   # range of lambdas to test 

for lam in lams:                        # for each lambda
    ridge.set_params(alpha=lam)         # set it as the hyperparameter
    ridge.fit(xtrain, ytrain)           # fit the ridge model with training data
    predictions = ridge.predict(xtest)  # test the model
    rmse = mean_squared_error(ytest, predictions, squared=False)    # compute rmse
    rmses.append(rmse)                                              # add rmse to the list

# find minimum rmse and optimal lambda
optimal_rmse = min(rmses)
optimal_lambda = lams[rmses.index(optimal_rmse)]

# plot
plt.plot(lams, rmses, linewidth=2)
plt.ticklabel_format(useOffset=False)
plt.xlabel('Lambda Value')
plt.ylabel('Root Mean Squared Error')
plt.axhline(optimal_rmse, color='red', linestyle='dashed', linewidth=2,label=f'Minimum RMSE: {optimal_rmse:.2f}')
plt.axvline(optimal_lambda, color='purple', linestyle='dashed', linewidth=2, label=f'Optimal Lambda: {optimal_lambda}')
plt.legend()
plt.title('Optimizing Lambda for Ridge Regression')
plt.show()

# set the model to be the one with a lambda value of 192
ridge.set_params(alpha=optimal_lambda)
ridge.fit(xtrain, ytrain)
predictions = ridge.predict(xtest)
rsq = ridge.score(xtest, ytest)

print(rsq - max_rsq)   # subtracting the r squared value from the model above

# the model has not improved very much, as it now accounts for about 1.7% more of the variability in popularity

#%% question 8 - pca

# making z-scored predictors array
predictors = np.column_stack((zduration, zdanceability, zenergy, zloudness, zspeechiness, zacousticness, zinstrumentalness, zliveness, zvalence, ztempo))

'''
1 - duration
2 - danceability
3 - energy
4 - loudness
5 - speechiness
6 - acousticness
7 - instrumentalness
8 - liveness
9 - valence
10 - tempo
'''

# correlation heatmap
pca_correlations = np.corrcoef(predictors, rowvar=False)
plt.imshow(pca_correlations)
plt.colorbar()
plt.title('Correlations Between Predictors')
plt.show()

# pca
pca = PCA().fit(predictors)                         # fitting pca model to the data
eigvals = pca.explained_variance_                   # ordered eigenvalues
loadings = pca.components_*-1                       # loadings matrix
rotated_data = pca.fit_transform(predictors)*-1     # og data in terms of the new coordinates

# screeplot
num_predictors = 10
x = np.linspace(1, num_predictors, num_predictors)
plt.bar(x, eigvals, color='lightpink')
plt.axhline(1, color='maroon')
plt.title('Scree Plot')
plt.xlabel('Principal Components')
plt.ylabel('Eigenvalues')
plt.show()

# graph to look at principal components
which_principal_component = 2
plt.bar(x, loadings[which_principal_component,:]*-1, color='maroon')
plt.xlabel('Factor')
plt.ylabel('Loading')
plt.title(f'Loadings Plot for PC {which_principal_component + 1}')
plt.show()

# using the elbow method, it looks like there is 1 principal component
# when the loadings are examined, it looks like "softness" or "calmness" 

# using the kaiser criterion, there are 3 principal components
# the 2nd pc looks like "melancholy"
# the 3rd has high speechiness and high liveness
# i'm not sure what i would call that, maybe 

# using the 90% variance method:
varexplained = eigvals/sum(eigvals)
threshold = .90
eigsum = np.cumsum(varexplained)
np.cumsum
print('number of factors to account for at least 90% variance:', np.count_nonzero(eigsum < threshold))

# the fourth pc is longness and slowness
# the fifth pc is upliftingness and fastness
# the sixth pc is longness and lack of instrumentalness
# based on these criteria, i would say there are 3 pc

print(f'variance accounted for with 3 principal components is {eigsum[2]* 100:.2f}%')

# clustering with the rotated data:
clusteringdata = np.column_stack((rotated_data[:,0], rotated_data[:,1], rotated_data[:,2]))

# silhouette method to determine number of clusters
silhouette_sums = np.empty([9,1])*np.nan    # initialize container to store sums

for ii in range(2, 11):                     # loop through each number of clusters
    kmeans = KMeans(n_clusters=ii).fit(clusteringdata)       # compute kmeans
    cid = kmeans.labels_                        # vector of cluster ids
    ccoords = kmeans.cluster_centers_           # coordinate loaction for centroids
    s = silhouette_samples(clusteringdata, cid) # compute mean silhouette coefficient of all samples
    silhouette_sums[ii-2] = sum(s)              # sum of all silhouette scores

plt.plot(np.linspace(2, 9, 9), silhouette_sums, color='lightseagreen')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Silhouette Scores')
plt.title('Determining the Optimal Number of Clusters')
plt.show()

# based on this, optimal clustering appears at k=2

clusters = 2
kmeans = KMeans(n_clusters = clusters).fit(clusteringdata)
cid = kmeans.labels_
ccoords = kmeans.cluster_centers_

for ii in range(clusters):
    plot_index = np.argwhere(cid == ii) # plotting the points where the index is the cluster of the iteration
    plt.plot(clusteringdata[plot_index,1], clusteringdata[plot_index,2], 'o', markersize=1)
    plt.plot(ccoords[ii-1,0], ccoords[ii-1,1], 'o', markersize=5, color='black')    # plot centroids
    plt.plot(ccoords[ii-2,0], ccoords[ii-2,1], 'o', markersize=5, color='black')
    plt.xlabel('PC2')
    plt.ylabel('PC3')
    plt.title('PC 2 vs PC 3')

#%% question 9 - predicting mode from valence

# the outcome (mode) is binary, so we will use logistic regression

# first we will make sure the labels are all 0 and 1 and plot them:
    
major_valence = major_tracks[:,16].astype(float)
minor_valence = minor_tracks[:,16].astype(float)
    
num_major = len(major_tracks)
avg_major_valence = np.mean(major_valence)
std_major_valence = np.std(major_valence)
print('number of tracks in a major key:', num_major)
print('average valence for tracks in a major key:', avg_major_valence)
print('standard deviation of valence for tracks in a major key:', std_major_valence)

num_minor = len(minor_tracks)
avg_minor_valence = np.mean(minor_valence)
std_minor_valence = np.std(minor_valence)
print('number of tracks in a minor key:', num_minor)
print('average valence for tracks in a minor key:', avg_minor_valence)
print('standard deviation of valence for tracks in a minor key:', std_minor_valence)

plt.scatter(valence, mode, s=1, color='black')
plt.xlabel('Valence Score')
plt.ylabel('Mode')
plt.yticks(np.array([0,1]))
plt.show()

# from this brief eda, it does not really look like there is a difference in valence score for major and minor

x = valence.reshape(len(data),1)
y = mode    
                  
xtrain, xtest, ytrain, ytest = train_test_split(x ,y, test_size=0.2, random_state=n_number)
model = LogisticRegression().fit(xtrain,ytrain)

predictions = model.predict(xtest)
conf_matrix = confusion_matrix(ytest, predictions)
print(model.score(x,y))

# from the confusion matrix we can see that the model only got 6528 right and it always predicted 1
# it was correct about 62% of the time, but this is not a good model

x1 = np.linspace(min(valence), max(valence), len(valence))
y1 = x1 * model.coef_ + model.intercept_
sigmoid = expit(y1)

plt.plot(x1, sigmoid.ravel(), color='r', linewidth=2)
plt.scatter(valence, mode, s=1, color='black')
plt.hlines(0.5, min(valence), max(valence), colors='gray', linestyles='dotted')
plt.xlabel('Valence Score')
plt.ylabel('Mode')
plt.yticks(np.array([0,1]))
plt.show()

# based on this, it does not look like we can predict mode from valence
# we will try to see if there is a better predictor

predictors = [popularity, duration, danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, tempo, time_signature]

for predictor in predictors:
    x = predictor.reshape(len(data),1)
    y = mode    
    xtrain, xtest, ytrain, ytest = train_test_split(x ,y, test_size=0.2, random_state=n_number)
    model = LogisticRegression().fit(xtrain,ytrain)
    predictions = model.predict(xtest)
    print(model.score(x,y))
    
# all of the models have the exact same score - they all always predict 1
# so we cannot predict mode from any of these predictors

#%% question 10 - predicting genre from pcs

# we will be predicting the genres from the three principal components identified above
# first we will map the genres to qualitative data

genrelist = []
for ii in data[:,19]:
    if ii not in genrelist:
        genrelist.append(ii)
    
genresnumerical = np.empty([len(genres),1])
for ii in range(len(genres)):
    index = genrelist.index(genres[ii])
    genresnumerical[ii] = index
    
X = np.column_stack((rotated_data[:,0], rotated_data[:,1], rotated_data[:,2]))
y = genresnumerical.reshape(len(genresnumerical),)

# fitting the tree model:

xtrain, xtest, ytrain, ytest = train_test_split(X ,y, test_size=0.2, random_state=n_number)

num_trees = 500
clf = RandomForestClassifier(n_estimators=num_trees, max_leaf_nodes=1000,max_depth=1000,).fit(xtrain, ytrain)

# predicting, getting accuracy
predictions = clf.predict(xtest)
model_accuracy = accuracy_score(ytest, predictions)
print('random forest model accuracy:', model_accuracy)

# plotting predictions vs actual:
plt.scatter(predictions, ytest, s=3, marker='o', color='darkorange')
plt.xlabel('Predicted Genre')
plt.ylabel('Actual Genre')
plt.title('Predictions vs Actual Values for Random Forest Model')

#%% extra

# splitting the data into subsets for each time signature
# and plotting their genres

tracksin5 = data[np.where(data[:,18] == 5)]
plt.hist(genresnumerical[tracksin5[:,0].astype(int)], bins=len(genrelist), color='cornflowerblue')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.title('Genres for Tracks with 5 Beats/Measure')
plt.show()

tracksin4 = data[np.where(data[:,18] == 4)]
plt.hist(genresnumerical[tracksin4[:,0].astype(int)], bins=len(genrelist), color='cornflowerblue')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.title('Genres for Tracks with 4 Beats/Measure')
plt.show()

tracksin3 = data[np.where(data[:,18] == 3)]
plt.hist(genresnumerical[tracksin3[:,0].astype(int)], bins=len(genrelist), color='cornflowerblue')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.title('Genres for Tracks with 3 Beats/Measure')
plt.show()

tracksin1 = data[np.where(data[:,18] == 1)]
plt.hist(genresnumerical[tracksin1[:,0].astype(int)], bins=len(genrelist), color='cornflowerblue')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.title('Genres for Tracks with 2 Beats/Measure')
plt.show()
