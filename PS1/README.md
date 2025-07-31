# cs229-assignments

## Problem Set 1

Only the solution of the sections tagged with `(coding problem)` in the assignment PDF are included.

### 1-b : Logistic Regression Classifier with Newton Method

Newton Method to converge faster than gradient descent using the Hessian(second derivative).

### 1-e : Gaussian Discriminant Analysis Model

GDA is a generative model tries to predict how the data was generated assuming each class follows a Gaussian distribution. 
It estimates Gaussian Distribution parameters mean and covariance for each class, then uses Bayes theorem to predict the class of a new input. 

### 2-c,d,e : Positive-Only and Unlabeled Learning

Show the difference of training between the true labels and positive-only labels using the logistic regression model we implemented in the 1-b 

### 3-d : Website Traffic Prediction Model using Poisson regression

Poisson distribution is a member of Exponential Family that is used to model counts, the number of times some event occurs during a fixed interval of time.

Exponential family is a family of distributions that can be represented in PDF(Probability Density Function) format for continous data. 

A Generalized Linear Model is a modeling approach that uses exponential family distributions to generalize linear regression to non-normal data.

In this problem, we are implementing a GLM Model to train on a non-normal data(website visitor count) and to model the mean using a Exponential Family Distribution(Poisson) appropriate to the data type(count).

If it was binary data, we could use the Bernoulli distribution or for data from real numbers, we could use the Gaussian directly.

### 5-b,c : Locally Weighted Regression

Use the distance between the test instance and the training instance as weight while computing the contribution of each training instance to theta

Uses the whole training set in each prediction