# Quora Question Pairs

### Problem Statement
As the knowledge-sharing platforms such as Reddit, Quoara, and Stack Exchange gain popularity, the need to identify previously asked questions is also getting more important. Same question can be asked with tens of different wording and getting more effective at identifying duplicate questions allow these platforms to offer a better user experience. Also contributers and seekers can spend less time on reduntant tasks. Another important use case for this feature is smart assistants' ability to understand a question even if it is worded differently everytime.

### Dataset and Project Description
For this project, I will be using Quora's data from a Kaggle competition.

The data can be downloaded from: https://www.kaggle.com/c/quora-question-pairs/data

In the dataset we are given question pairs on each row with a label indicating whether those questions have the same intent or not. If the 'is_duplicate' column is 0, it means that the question pair don't have the same intent and if it is 1, it means that they have the same intent. We are expected to create a model and predict whether the questions on the test data are duplicate or not.

### Solution Approach
In order to tackle this problem I've divided my approach in small tasks and they can be summed up in a pipleine as follows:
- Data Exploration/Analysis
- Feature Engineering
- Machine Learning Model Implementation
  - Algorithm choice
  - Hyperparameter Optimisation
  - Measures to prevent overfitting
 
#### Data Exploration/Analysis
In my opinion this is by far the most important stage of any machine learning project. If we understand the data that we are dealing with at a very deep level, we can handle missing values and outliers in an optimal way, we can generate more effective features, we can build suitable models based on the data and the results of our model would reflect this deep understanding.  
Also this the stage I make sure the train and test data are coming from same distribution. Also I decide on what kind of a validation model I will be using and what are the canditate features for the feature engineering part.

In an nlp project like this, after investigating the size of the data, outliers, missing values and key features, it is a good practice to compare some features of the train and test data to make sure that our model can generalise well on the test data as well. I compared the character counts and word counts in the questions of train and test data and plot histogram graphs.
![character count](/images/char_count.png)
![word count](/images/word_count.png)
As we can see from the histograms, both the character counts and the word counts are very close to each other which means we don't have to worry about this in terms our model's ability to generalise well on the test data.

Another very useful visualisation tool that I use quite often in nlp projects is word cloud. It allows me to recognise patterns and biases in the word corpus. The word cloud for the train data is as follows:
![word cloud](/images/wordcloud.png)
With a quick glance at the word cloud, we can detect 4 concepts dominating the questions: India, US, tech, money..

#### Feature Engineering
This is the stage where we put our findings from the data exploration part into practice. In order to get the best out of data, we need to iteratively generate features and test their usefulness. For this project, I focused on generating features that are high in quality and low in quantity. This approach also helps with the interpretability of our model. The features I generated was based on shared words between questions, shared substrings between questions, tf-idf, Levensthein distance and some meta features that utilise the question frequency in the dataset. The details of these features and their implementation along with the entire code are in the jupyter notebook file called quora_notebook.ipynb

#### Machine Learning Model Implementation
Implementing an effective machine learning model require to make some important decisions in terms of choosing the right algorithm for the task, optimasing the hyperparameters of the model and making sure we are not overfitting.
- After implementing naive bayes and logistic regression models for this project, I got better results from gradient boosting machine and that's why I implemented it using XGBoost framework. 
- I optimised the hyperparameters via random grid search using the sci-kit learn library's RandomizedSearchCV function. Random search is almost as effective as grid search but takes a lot less time especially when we are dealing with large datasets. Although it is not guarenteed that random search will find the optimal hyperparameters, it usually finds almost optimal solutions.
- To make sure my model was not overfitting and will generalise well, I used early stopping with 10 rounds and also made sure that train and validation loss are close to each other.

### Takeaways
In this project, I achieved decent results with relatively little effort in feature engineering and model building. I put the most emphasis on the data exploration part. Validation logarithmic loss of my model was around 0.16 which is very impressive for a relatively light-weight model(i.e. No ensembling, not using deep learning, small number of features). 

Results can be improved through more extensive feature generation that includes more in depth data pre and post processing,using algorithms such as word2vec or doc2vec and getting embeddings, and using more advanced models such as RNN with LSTM. 

However this project proves that with a careful data exploration, we can generate quick and efficient models that achieve pretty good results with real world use-cases.

Understanding the data that we are dealing with is far more important than the tools we use.

