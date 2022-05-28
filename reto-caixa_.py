#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importacion de librerias
import pandas as pd
from datetime import datetime
import string
import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier


# In[2]:


# Carga de data
df = pd.read_csv('../data/train.csv')
df_tweets = pd.read_csv('../data/tweets_from2015_#Ibex35.csv')
df_test = pd.read_csv('../data/test_x.csv')


# In[3]:


df


# In[4]:


# Datos nulos de la data historica
df.isnull().sum()


# In[5]:


# Asignamos la fecha como nuestro indice
df.Date  = pd.to_datetime(df["Date"])
df.set_index(['Date'], inplace=True)


# In[6]:


df


# In[7]:


df_tweets.handle.value_counts().head(20)


# In[8]:


# Datos nulos de la data historica de twitter
df_tweets.isnull().sum()


# In[9]:


df_tweets


# In[10]:


# Removemos los datos nulos
df_tweets = df_tweets.dropna()


# In[11]:


df_tweets['Date'] = pd.to_datetime(df_tweets['tweetDate'],errors='coerce')


# In[12]:


df_tweets.dtypes


# In[13]:


df_tweets.Date = df_tweets.Date.apply(lambda x: str(x)[0:10])


# In[14]:


df_tweets


# In[15]:


df_tweets.Date = df_tweets.Date.astype('datetime64[D]')


# In[16]:


df_tweets.dtypes


# In[17]:


# Hacemos un merge de la data que contiene los tweets
# y la data sin tweets
df_merged = df.merge(df_tweets,on='Date')


# In[18]:


df_merged


# In[19]:


# Valores nules de la data mezclada
df_merged.isnull().sum()


# In[20]:


df_merged.Date[0], df_merged.Date[4302]


# In[21]:


# lista de las stopwords en español
stopwords = nltk.corpus.stopwords.words("spanish") + list(string.punctuation)
stopwords


# In[22]:


# creamos la funcion para remover 'palabras' con una longitud menor a uno
def remove(text): 
    return ' '.join(word for word in text.split() if len(word)>1)


# In[23]:


# definimos la funcion para tokenizar nuestro corpus
def stop_words_token(row):
    
    word_tokens = word_tokenize(row)

    return [w for w in word_tokens if not w in stopwords]


# In[24]:


spanish_punctuations = string.punctuation
punctuations_list = spanish_punctuations
def cleaning_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)


# In[25]:


# removemos las urls del corpus
def cleaning_URLs(data):
    return re.sub('https://','',data)


# In[26]:


# removemos los numeros del corpus
def cleaning_numbers(data):
    return re.sub('[0-9]+', '', data)


# In[27]:


# aplicamos la funcion remove al dataset
df_merged['text_clean'] = df_merged['text'].apply(remove)


# In[28]:


df_merged['text_clean'] = df_merged['text_clean'].str.replace('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ')


# In[29]:


# aplicamos la funciones cleaning_punctuations, cleaning_URLs y cleaning_numbers a la data
df_merged['text_clean']= df_merged['text_clean'].apply(cleaning_punctuations)
df_merged['text_clean'] = df_merged['text_clean'] .apply(cleaning_URLs)
df_merged['text_clean'] = df_merged['text_clean'] .apply(cleaning_numbers)


# In[30]:


# definimos nuestro analizador de sentimiento
sid = SentimentIntensityAnalyzer()


# In[31]:


# aplicamos el analizador de sentimientos a la data
df_merged['sentiment'] = df_merged.text_clean.apply(sid.polarity_scores)


# In[32]:


# creamos una listas en las filas de la columna 'sentiment'
df_sentiment = pd.DataFrame(df_merged['sentiment'].tolist())


# In[33]:


df_merged = df_merged.join(df_sentiment)


# In[34]:


# dataframe con las features: neg, pos, neu y compound
df_merged


# In[35]:


df_merged['compound'][df_merged['compound'].sort_values() == 0].shape


# In[36]:


df_merged


# In[37]:


# Eliminamos las variables menos relevantes para el entrenamiento del modelo
to_drop = ['text', 'text_clean', 'tweetDate', 'handle', 'sentiment',
          'neg', 'neu', 'pos']
df_clean = df_merged.drop(to_drop, axis=1)
df_clean


# In[38]:


# eliminamos valores nules
df_clean = df_clean.dropna()


# In[39]:


df_clean


# In[40]:


# indexamos la columna Date
df_clean.set_index(['Date'], inplace=True)


# In[41]:


df_clean.index


# In[42]:


# hacemos un groupby y luego promediamos las columnas por cada dia
df_clean = df_clean.loc[:, ['Open', 'High', 'Low', 'Close','Adj Close', 'Volume',
                            'Target', 'compound']].groupby(df_clean.index).mean()


# In[43]:


df_clean


# In[44]:


df_clean.isnull().sum()


# In[45]:


# declaramos las variables predictoras y la variable a predecir
X = df_clean.drop(['Target'], axis=1)
y = df_clean['Target']
# train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# In[46]:


# escalamos la data para que todas las columnas tengan la misma desviacion estandar
# y la misma media
def scaled(data):
    columns_names = data.columns 
    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data))
    return data


# In[47]:


X_scaled = scaled(X)


# In[48]:


X_scaled


# In[49]:


# seleccionamos la data que no contiene twitter para entrenar el modelo
df_train = df.loc[df.index < '2015-01-05']
df_train


# In[50]:


# numero de valores nulos de la data que no contiene tweets
df_train.isnull().sum()


# In[51]:


# eliminamos los valores nulos de la daa que no contiene tweets
df_train = df_train.dropna()
df_train


# In[52]:


df_train.isnull().sum()


# In[53]:


# declaramos las variables predictoras y la variable a predecir en la data que 
# no contiene tweets
X_notweets_train = df_train.drop(['Target'], axis=1)
y_notweets_scale = df_train.Target


# In[54]:


# escalamos la data
X_scaled_nontweets = scaled(X_notweets_train)


# In[55]:


X_scaled_nontweets


# In[56]:


# definimos nuestro modelo
model = LinearRegression()


# In[57]:


# entrenamos el modelo sobre la data que no contiene tweets
model.fit(X_scaled_nontweets, y_notweets_scale)


# In[58]:


# entrenamos el modelo sobre la data que no contiene tweets
model.fit(X_train, y_train)


# In[59]:


# prediccion
y_pred = model.predict(X_test)


# In[60]:


# asignamos 1 y 0 a la variable predecida
# ya que se trata de una clasificacion binaria
y_pred = np.where(y_pred > 0.5, 1, 0)


# In[61]:


# medimos el f1 score de la data de prueba y la data de entrenamiento
f1_score = metrics.f1_score(y_test, y_pred, average='macro')
f1_score


# In[62]:


df_test


# In[63]:


# indexamos la columna 'Date' en la data de evaluacion 
df_test.set_index(['Date'], inplace=True)


# In[64]:


# escalamos la data de evaluacion
df_test_scale = scaled(df_test)


# In[65]:


df_test_scale


# In[66]:


# prediccion sobre la data de evaluacion
y_true = model.predict(df_test_scale)
prediction = np.where(y_true > 0.5, 1, 0)
prediction


# In[67]:


prediction


# In[74]:


output = pd.DataFrame(columns=['test_index', 'target'])
output['target'] = prediction
output['test_index'] = df_test.index
output.to_csv('../data/y_pred.csv', header=True, index=False)
output.to_json('../data/y_pred.json', orient="split")


# In[ ]:




