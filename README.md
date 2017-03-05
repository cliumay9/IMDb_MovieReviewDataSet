# IMDb_MovieReviewDataSet
Conducting Sentiment Analysis on the IMDb Movie review dataset.
We First approach this problem with a classic bag-of-words model; 
we use gridsearch algorithm to fit the dataset into a logistic regression model. 
Then, we realize this algorithm is very computational heavy so we use out-of-core learning model
For this reason, we use our second approach, out-of-core learning model. 

In conclusion, despite the accuracy decreases for using out-of-core learning, 
the run time significantly lowers as it took less than 1 minute while the classic approach takes about 30 minutes.
