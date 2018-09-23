# forest_cover
Kaggle Competittion for Predicting Forest Cover Type


## Intro

In this kaggle competition, participants are asked to predict the cover type
of trees based on geographic features such distance from water, elevation,
sun exposure, soil type, etc.

I chose this competition because it was a great opportunity to practice
my model building and advanced data science skills. The dataset provided is
very clean and already pre-processed. This allows me to focus on exploration
the data, feature engineering, feature selection, model tuning, and ensembling.

## Results

Through feature engineering, model tuning, and ensembling, I acheived an
accuracy score of 79%, which ranks in the top 15% of the kaggle leader board.

The most effective model was a stacked model that ran the predictions
from a number of base models through the LightGBM algorithm.



## Pipeline


#### 1) Clean_Explore

The first step is in `Clean_Explore.ipynb`. Here I familiarized myself with
the dataset, looked for any opportunities to clean it (eg were there anomolies? missing values?),
and examined feature distributions to prepare for modeling.


#### 2) Feature_Engineering

In `Feature_Engineering.ipynb` I performed some additional preprocessing of the data
such as scaling the data, and tried to come up with creative ways to combine
the features into new, predictive features. For example, I used PCA to create
linear combinations of the original features, clustered using KMeans, and
generated class probability vectors using NaiveBayes.

This created a large set of new custom features which I then ran through
a feature selection process stored in `feature_selector.py` (see below).
This narrowed down the feature set to a more manageable number of only the
most relevant features in the dataset.

Finally, I created polynomial combinations of the features to generate interaction
terms that would be even more predictive of Cover_Type and re-ran the newly generated
features through the feature selection process.

#### 2.5) Feature Selection

Datasets with increasingly large number of features start to cause problems
due to the curse of dimensionality. Not only do models become more complex, and take
longer to run, but they often suffer in performance due variables that capture
false patterns in noise rather than the true pattern of the dependent variable.

I combined a few different feature selection techniques to select only the most
relevant features to my model.

__Top Correlated__
  - Select the features with highest correlation to dependent.

__Top Chi2__
  - Use chi2 test to test dependence and select highest scored features.

__Recursive Feature Elimination__
  - Iteratively build models and at each step remove the least informative features
  in the model.

__Lasso__
  - Use L1 regularization to reduce model complexity by zeroing out extraneous features.

I then scaled all the results form the above tests and selected the features
with the highest combined score.

#### 3) Modeling


After establishing my final feature set, I began modeling by applying and tuning
a number machine learning algorithms to the final dataset. You can see the step
by step process in `Modeling.ipynb`.

For algorithms with a manageable number of parameters, I used a grid search and
cross validation to find the optimal parameters for each algorithm. For
algorithms with larger number of parameters (such as Ensembled and Boosted Trees),
I used bayesian optimization to explore the parameter space and approximate
the maximum in the posterior of the cross-validated model scores. This reduced
training times while still producing well-tuned models.

The algorithms I used were:

  - Logistic Regression
  - Lindear Discriminant Analysis
  - K-Nearest Neighbors
  - Support Vector Machines
  - Random Forest, and Extra Random Tree Ensembles
  - Multi-layer Perceptrons
  - The LightGBM and XGBoost implementations of Gradient Boosted Decision Trees

#### 4) Ensembling and Stacking

After training each individual model, I further increased my predictive accuracy
by ensembling their predictions together and took the majority vote across predictors.

Ensembling in this way is an easy way to build stronger models that are better at
generalizing to the dataset without having to do additional training and a
negligible amount of coding time.

I also performed a more advanced version of ensembling known as stacking.
Stacking involves training "Meta-Estimators" on the predictions of your
base estimators, thus allowing your models to learn from each other.

My best performing model reached an accuracy of 79%, which put me in the top 15%
of the kaggle leaderboard.
