<b>Data Mining Project - Who Made That Movie
==================================

Data Mining project using IMDb, Movilens and Wikipedia datasets

https://github.com/iaperez/DataMiningProject-WhoMadeThatMovie

<b>Project Members
  
 *  Aisha Kigongo
 *  Ignacio Pérez

<b>Project Statement

We are interested in using the ratings that we found in the Movielens, and the information available in IMDb and Wikipedia to predict the success of movies. The main idea is to quantify the trajectory and the experience of the people behind movies, and then apply the resulting metric to estimate the success of the movies that are in post-production stage.

First, we need to define a metric to measure the success of a movie. In particular, we are considering two main areas: movies that have a high number of viewers (income success) and movies that have high rates (ratings) from the viewers (quality success). Considering that the number of viewers (related with income success) could be associated with the budget of the movie and other sociocultural factors, we prefer to define our success metric for movies by using the ratings: a more successful movie will receive better ratings from the users.

To quantify the trajectories of people behind movies, we will use the ratings and categories of the movies in which those people have worked. That metric will be represented in a vector that indicates the experience or success of people in specific types of movies: drama, action, comedy, etc. For example, if we consider a vector with comedy, drama and action categories, and famous people like Steven Spielberg,  Bruce Willis and Jim Carrey, the vectors for them should look like:


    |                  | Action | Comedy | Drama |
    |Jim Carrey        |   0.23 | 0.72   | 0.12  |
    |Steven Spielberg  |   0.52 | 0.01   | 0.62  |
    |Bruce Willis      |   0.75 | 0.22   | 0.13  |
 
In the vector of experience, we need to consider to normalize by the number of movies that each actor has worked.

Now, given the trajectories of the people that are behind a movie, we need forecast a rating from a particular user. Here we can compare the vector of preferences of the user (by category) with an aggregated vector of experience by category of the people behind that movie. The movie vector should consider the role of each person (a director of a movie and his or her experience should be more important than the experience of a camera man)

By comparing the aggregated experience vector of the people working in the movie and the rating vector of ratings of each user, we will have an idea of the possible rating. We still need to work in a way to join this two vectors and get a forecasting for the rating.

<b>Datasets
http://www.grouplens.org/node/73
http://www.imdb.com/interfaces

<b>Data Analysis

There are some questions that we are interested to investigate during the project:

1. The mean, median, standard deviation can be used to find the distribution of user ratings for the movies in the dataset. 
2. Are documentaries better than horrors? 
3. Have movies improved over time? are old movies better than new ones given that there many factors that influence movies today? 
