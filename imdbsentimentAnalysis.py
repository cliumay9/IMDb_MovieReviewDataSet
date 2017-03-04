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

import numpy as np
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
df('review') = df['review'].apply(preprocessor)                

######## Employing bag-of-words model ###########
'''We use bag-of-words model to act as a way to preprocess. That is, representing text as numerical
feature vectors '''

'''
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer()
docs = np.array([
                 'The sun is shining',
                 'The weather is sweet',
                 'The sun is shining and the weather is sweet'])
bag = count.fit_transform(docs)'''








