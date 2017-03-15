# Sentiment Analysis IMDb movie 
import theano
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

## Bag-of-Words Approach ##

######## Employing bag-of-words model ###########
'''Turn Text into Feature Vectors
We use bag-of-words model to act as a way to preprocess.  '''

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
    text = re.sub('<[^>]*>', '', text)
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
# Fitting the training data into our gridsearch
gs_lr_tfidf.fit(X_train,Y_train)

print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_) #Note: 89.2%
##
# Now apply our grid search LR model to predict test result and and determine its accuracy
clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.3f' % clf.score(X_test, Y_test)) # 89.7%

## Alternate Approach ##
'''
In order to make the algorithm to be less computation heavy, we stream the data
This is called Out-of-core learning
'''
import numpy as np
import re
from nltk.corpus import stopwords
stop = stopwords.words('english')
    
'''Define a funciton tokenizer that removes regular expressions
 but not emojis while removing stopwords with stopwords library'''
def tokenizer(text):
    text =re.sub('<[^>]*>', '', text)
    emojis = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+',' ', text.lower()) +\
        ' '.join(emojis).replace('-','') 
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

'''Define a stream document funcction that read a document and return a doc'''
def stream_docs(path):
    with open(path, 'r') as csv:
        next(csv) #Skipping the header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label

# To test our stream_docs function, next(stream_docs(path = './movie_data.csv'))

# Define a function that stream document with specified 'size'
def get_minibatch(doc_stream, size):
    docs, y =[],[]
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y

'''We cannot use CountVectorizer and Tfidfvectorizer 
as they require to hold either the complete vocabulary or keeping
all feature vecotrs in memory to calculate the counterpart
Thus, we use HashingVectorizer since it is data-independent
(This algorithm use hashing trick via 32bit MurmurHash3 Algorithm,
sites.google.com/site/murmurhash)
'''            
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
vect = HashingVectorizer(decode_error = 'ignore',
                         n_features = 2**21,
                         preprocessor = None,
                         tokenizer = tokenizer)
clf = SGDClassifier(loss = 'log', random_state = 0, n_iter=1)
doc_stream = stream_docs(path= './movie_data.csv')
# Note: Increasing n_features avoid hash collision 
# But increase number of coefficients in the logistic regression model
'''
Out-of-core learning executing
'''
## Use training set to train our out-of-core learning algorithm
import pyprind
pbar = pyprind.ProgBar(45) #To set up a bar for the estimated time for our algorithm
classes = np.array ([0,1])
for _ in range(45):
    X_train, Y_train = get_minibatch(doc_stream, size =1000)
    # exceuting 45 minibatches and each batch contains 1000 documents
    if not X_train:
        break
    X_train = vect.transform(X_train) # Transforming our X_train to our SGDC classifier with Hashing
    clf. partial_fit(X_train, Y_train, classes = classes)
    # Fitting (partially) with our training set to our model
    pbar.update()
# Use the test set to evaluate our model's performance
X_test, Y_test = get_minibatch(doc_stream, size = 5000)
X_test = vect.transform(X_test)
print('Accuracy %.3f' % clf.score(X_test, Y_test)) #86.6%

# use the test test to update our model
clf = clf.partial_fit(X_test, Y_test)




