# Sentiment Analysis IMDb movie 

import pyprind
import pandas as pd
import os
'''Movie Revieww data set was downloaded from http://ai.stanford.edu/~amaas/data/sentiment/
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}

  '''

# Turn decompressed dataset into a single CSV file
'''Use archiver to extract files from the download archive
  OR
  execute tar -zxfaclImdb_v1.tar.gz (FOR MAC AND LINUX)'''
############### Turning our downloaded dataset into csv file ################
### Data Preprocessing ###

pbar = pyprind.ProgBar(50000)
labels = {'pos':1,'neg':0}
df = pd.DataFrame()
for s in ('test','train'):
    for l in ('pos', 'neg'):
        path = '/Users/calvinliu/Downloads/aclImdb/%s/%s' % (s,l)
        for file in os.listdir(path):
            
            with open(os.path.join(path, file), 'r',
                      encoding = 'utf-8') as infile:  #Set encoding = 'utf-8'

                txt = infile.read()
            df = df.append([[txt, labels[l]]], ignore_index = True)
            pbar.update()
             
df.columns = ['review', 'sentiment']
# Note that df is well sorted based on how the file was extracted; we will shuffle df

######## Employing bag-of-words model ###########
'''We use bag-of-words model to act as a way to preprocess. That is, representing text as numerical
feature vectors '''

### Transform words into feature Vectors
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
count = CountVectorizer()
docs = np.array(['The sun is shining','The weather is sweet',
                 'The sun is shining and the weather is sweet'])
bag = count.fit_transform(docs)

### Assessing word relvancy with term frequency-inverse document frequency
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
np.set_printoptions(precision =2)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())


np.random.seed(0) #Set seed to 0 to have same consistency
df = df.reindex(np.random.permutation(df.index))
df.to_csv('./movie_data.csv', index = False)
# DF = pd.read_csv('./movie_data.csv')  # Make sure the movie_data csv file is readable

##### Cleaning Data #####

'''Define a function to clean data.'''
import re
def preprocessor(text):
    text =re.sub('<[^>]*>', '', text)
    emojis = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+',' ', text.lower()) +\
        ' '.join(emojis).replace('-','')
    return text
    
# To test our preprocessor function work
# res = preprocessor("</a>This :) is :( a test :-)!")
# print('Preprocessor on "</a>This :) is :( a test :-)!":\n\n', res)
df['review'] = df['review'].apply(preprocessor)                

### Turn df to tokens with tokenizer_porter function

'''Define two function to tokenize with normal text.split and Porter Stemmer Algorithm, respectively. 
Also, other algorithms like Snowball Stemmer or Lancaster Algorithm
(http://www.nltk.org/api/nltk.stem.html)
Use pip install nltk package'''
# tokenizer vs tokenizer_porter

def tokenizer(text):
    return text.split()
    
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
    x=[]
    for word in text.split():
        x.append(porter.stem(word))
    return x
# to test our tokenizer function
# tokenizer_porter('runners like running and thus they run')

### Remove stop words which are avaialable from the NLTK library
# Stopwords
import nltk
nltk.download('stopwords')  #Obtain stopwords from the library

from nltk.corpus import stopwords
stop = stopwords.words('english')

def removeStop(text):   
    l = []
    tp = tokenizer_porter(text)
    for w in tp:
        if w not in stop:
            l.append(w)
    return l
removeStop('a runner likes running and runs a lot')

# [w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:] if w not in stop]
# other way to write removeStop(text)

'''Training a logistic regression model for document classification.'''
# Divide the dataframe of cclean text documents into 25k for training and 25k testing
X_train = df.loc[:25000, 'review'].values
Y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
Y_test = df.loc[25000:, 'sentiment'].values

# Use GridSearch to find the optimal set of parameters for the logistic regression model
# using 5-fold stratified cross-validation
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(strip_accents = None, lowercase = False,
                        preprocessor = None)

param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              ]
#Can add Third dictionary to see better determine which parameters are good for our Logistic regression model

# Logistic Regression - Term Frequency-Inverse Document Frequency
lr_tfidf = Pipeline([('vect',tfidf),
                     ('clf', LogisticRegression(random_state = 0))])
# Apply our set gridsearch to our logistic regression model
gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                           scoring = 'accuracy',
                           cv=5, verbose =1, 
                           n_jobs = -1)
# cv = number of fold
gs_lr_tfidf.fit(X_train,Y_train)

print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_) #Note: 89.2%
##
# Now apply our grid search LR model to predict test result and and determine its accuracy
clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.3f' % clf.score(X_test, Y_test)) # 89.7%


