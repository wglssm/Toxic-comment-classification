#import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from random import seed
from random import randint
import re
import nltk
from gensim.parsing.preprocessing import remove_stopwords
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation,Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D,SpatialDropout1D
from keras.models import Model

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


#load data
comment_train = pd.read_csv("train.csv")
comment_test=pd.read_csv("test.csv")
comment_train.head()
#random select some comment to see
seed(120)
# generate some integers and view the comments 
for _ in range(5):
    value = randint(0, 159571)
    print(comment_train["comment_text"][value])
    print('\n')


# EDA
x=comment_train.iloc[:,2:].sum()
ax= sn.barplot(x.index, x.values)
plt.title("# per class")
plt.ylabel('# of Occurrences')
plt.xlabel('Type ')
#adding the text labels
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
plt.show()


# In[70]:


text_length=[]

for i in range(len(comment_train)):
    text_length.append(len(comment_train["comment_text"][i].split()))
plt.hist(text_length,bins=40,range=(1,400))
plt.xlabel('length of words in comments')
plt.ylabel('# of observations')
plt.title('Histogram of length of words in comments')
plt.grid(True)
plt.show()


# In[112]:


#clean the comments

#to lower cases
comment_train["comment_text"] = comment_train["comment_text"].apply(lambda x: x.lower())
comment_test["comment_text"] = comment_test["comment_text"].apply(lambda x: x.lower())
(#Remove, Non-ASCI, and, Remove, URLs)
def remove_URL(text):
    return re.sub(r"https?://\S+|www\.\S+", "", text)
comment_train["comment_text"] = comment_train["comment_text"].apply(lambda x: remove_URL(x))
comment_test["comment_text"] = comment_test["comment_text"].apply(lambda x: remove_URL(x))

def remove_non_ascii(text):
    return re.sub(r'[^\x00-\x7f]',r'', text)
comment_train["comment_text"] = comment_train["comment_text"].apply(lambda x: remove_non_ascii(x))
comment_test["comment_text"] = comment_test["comment_text"].apply(lambda x: remove_non_ascii(x))

def remove_punct(text):
    return re.sub(r'[]!"$%&\'()*+,./:;=#@?[\\^_`{|}~-]+', "", text)
comment_train["comment_text"] = comment_train["comment_text"].apply(lambda x: remove_punct(x))
comment_test["comment_text"] = comment_test["comment_text"].apply(lambda x: remove_punct(x))

#remove \n
comment_train["comment_text"] = comment_train["comment_text"].map(lambda x: re.sub("\n", " ", x))
comment_test["comment_text"] = comment_test["comment_text"].map(lambda x: re.sub("\n", " ", x))

# remove stop words
#filtered_sentence = remove_stopwords(text)
comment_train["comment_text"] = comment_train["comment_text"].apply(lambda x: remove_stopwords(x))
comment_test["comment_text"] = comment_test["comment_text"].apply(lambda x: remove_stopwords(x))


# In[113]:


seed(120)
# generate some integers and view the comments 
for _ in range(5):
    value = randint(0, 159571)
    print(comment_train["comment_text"][value])
    print('\n')

# In[24]:

#split testing and training data
comment_train_data=comment_train['comment_text']
comment_train_target=comment_train[['toxic','severe_toxic','obscene','threat','insult','identity_hate']]
x_train,x_test,y_train,y_test = train_test_split(comment_train_data,comment_train_target,test_size=0.25, random_state=404)

# In[79]:


# build the baseline model :Naive Bayes model
cat=['toxic','severe_toxic','obscene','threat','insult','identity_hate']
max_f = 20000
model_NB = make_pipeline(TfidfVectorizer(max_features=max_f), MultinomialNB())
pred_test_nb = np.zeros((x_test.shape[0],len(cat)))

#fit the model for each label
for i,x in enumerate(cat):
    print(x,": ")
    model_NB.fit(x_train, y_train[x])
    pred_test_nb[:,i] = model_NB.predict(x_test)
sum_score=0
#get result
for i,x in enumerate(cat):
    print(x," Testing AUC:",metrics.roc_auc_score(y_test[x], pred_test_nb[:,i]))
    sum_score=sum_score+metrics.roc_auc_score(y_test[x], pred_test_nb[:,i])
print("average AUC score: ",sum_score/6)


# In[81]:


#Logistic Regression model
model_Log = make_pipeline(TfidfVectorizer(max_features=max_f), LogisticRegression()) 
pred_test = np.zeros((x_test.shape[0],len(cat)))
#fit the model for each label

for i,x in enumerate(cat):
    print(x,": ")
    model_Log.fit(x_train, y_train[x])
    pred_test[:,i] = model_Log.predict_proba(x_test)[:,1]
sum_score_log=0
#get auc score
for i,x in enumerate(cat):
    print(x," Testing AUC:",metrics.roc_auc_score(y_test[x], pred_test[:,i]))
    sum_score_log=sum_score_log+metrics.roc_auc_score(y_test[x], pred_test[:,i])
print("average AUC score: ",sum_score_log/6)                                               


# In[93]:


#submission
preds = np.zeros((len(comment_test), len(cat)))
for i ,j in enumerate(cat):
    model_NB.fit(x_train, y_train[j])
    labels = model_NB.predict(comment_test["comment_text"])
    preds[:,i] = labels
submid = pd.DataFrame({'id': comment_test["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = cat)], axis=1)
submission.to_csv('submission.csv', index=False)


# In[26]:


## how big is each word vector 
embed_size = 50 
# max number of words in a comment to use
maxlen = 200 
#tokenizer to the comments
tokenizer = Tokenizer(num_words=max_f,char_level=True)
tokenizer.fit_on_texts(list(comment_train_data))
tokenized_train=tokenizer.texts_to_sequences(comment_train_data)
tokenized_test=tokenizer.texts_to_sequences(x_test)
tokenized_submmit=tokenizer.texts_to_sequences(comment_test["comment_text"])
train_x=pad_sequences(tokenized_train,maxlen=maxlen)
test_x=pad_sequences(tokenized_test,maxlen=maxlen)
submmit_x=pad_sequences(tokenized_submmit,maxlen=maxlen)

#lstm model
inp = Input(shape=(maxlen,))
x = Embedding(max_f,embed_size)(inp)
x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = Dense(64, activation="relu")(x)
x = GlobalMaxPool1D()(x)
x= Dropout(0.1)(x)
x= Dense(6,activation="sigmoid") (x)

model_dp = Model(inputs=inp, outputs=x)
model_dp.compile(loss='binary_crossentropy',optimizer= 'adam',metrics=['accuracy'])
print(model_dp.summary())


# In[27]:

#fit the model
model_dp.fit(train_x,comment_train_target,epochs=6,batch_size=32,validation_split=0.1)


# In[28]:

#see the result
y_pred=model_dp.predict(submmit_x,batch_size=32)
submid = pd.DataFrame({'id': comment_test["id"]})
submission = pd.concat([submid, pd.DataFrame(y_pred, columns = cat)], axis=1)
submission[["toxic","severe_toxic","obscene","threat","insult","identity_hate"]]=y_pred
submission.to_csv('submission.csv',index=False)


# In[84]:


# 1d-Conv layer 

inp = Input(shape=(maxlen,))
x = Embedding(max_f,embed_size)(inp)
x = Conv1D(64, kernel_size=3, padding='valid', kernel_initializer='glorot_uniform')(x)
x = Dense(64, activation="relu")(x)
x = GlobalMaxPool1D()(x)
x= Dropout(0.1)(x)
x= Dense(6,activation="sigmoid") (x)
model_C = Model(inputs=inp, outputs=x)
model_C.compile(loss='binary_crossentropy',optimizer= 'adam',metrics=['accuracy'])
print(model_C.summary())


# In[85]:

#fit the model
model_C.fit(train_x,comment_train_target,epochs=6,batch_size=32,validation_split=0.1)


# In[86]:


y_pred_con=model_C.predict(submmit_x,batch_size=32)
submid = pd.DataFrame({'id': comment_test["id"]})
submission = pd.concat([submid, pd.DataFrame(y_pred_con, columns = cat)], axis=1)
submission[["toxic","severe_toxic","obscene","threat","insult","identity_hate"]]=y_pred_con
submission.to_csv('submission.csv',index=False)


# In[104]:


#lstm +Con_1 layer 
inp = Input(shape=(maxlen,))
x = Embedding(max_f,embed_size)(inp)
x = SpatialDropout1D(0.35)(x)
x = Conv1D(64, kernel_size=3, padding='valid', kernel_initializer='glorot_uniform')(x)
x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(64, activation="relu")(x)
x= Dropout(0.1)(x)
x= Dense(6,activation='sigmoid') (x)
model_lstm_c = Model(inputs=inp, outputs=x)
model_lstm_c.compile(loss='binary_crossentropy',optimizer= 'adam',metrics=['accuracy'])
print(model_lstm_c.summary())


# In[109]:
model_lstm_c.fit(train_x,comment_train_target,epochs=10,batch_size=16,validation_split=0.1)
# In[99]:


y_pred_con_l=model_lstm_c.predict(submmit_x,batch_size=32)
submid = pd.DataFrame({'id': comment_test["id"]})
submission = pd.concat([submid, pd.DataFrame(y_pred_con_l, columns = cat)], axis=1)
submission[["toxic","severe_toxic","obscene","threat","insult","identity_hate"]]=y_pred_con_l
submission.to_csv('submission.csv',index=False)


#Tuning 
#batch size
batch_16=model_lstm_c.fit(train_x,comment_train_target,epochs=10,batch_size=16,validation_split=0.1)
batch_32=model_lstm_c.fit(train_x,comment_train_target,epochs=10,batch_size=32,validation_split=0.1)
batch_64=model_lstm_c.fit(train_x,comment_train_target,epochs=10,batch_size=64,validation_split=0.1)
batch_128=model_lstm_c.fit(train_x,comment_train_target,epochs=10,batch_size=128,validation_split=0.1)
batch_256=model_lstm_c.fit(train_x,comment_train_target,epochs=10,batch_size=256,validation_split=0.1)
#plot for batch size difference
plt.suptitle('batch size difference', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.plot(batch_16.history['val_loss'], color='b', label='Validation Loss_16')
plt.plot(batch_32.history['val_loss'], color='r', label='Validation Loss_32')
plt.plot(batch_64.history['val_loss'], color='g', label='Validation Loss_64')
plt.plot(batch_128.history['val_loss'], color='y', label='Validation Loss_128')
plt.plot(batch_256.history['val_loss'], label='Validation Loss_256')
plt.legend(loc='upper right')


#drop out rate difference 
#0.1
inp = Input(shape=(maxlen,))
x = Embedding(max_f,embed_size)(inp)
x = SpatialDropout1D(0.35)(x)
x = Conv1D(64, kernel_size=3, padding='valid', kernel_initializer='glorot_uniform')(x)
x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(64, activation="relu")(x)
x= Dropout(0.1)(x)
x= Dense(6,activation='sigmoid') (x)
model_lstm_c = Model(inputs=inp, outputs=x)
model_lstm_c.compile(loss='binary_crossentropy',optimizer= 'adam',metrics=['accuracy'])
drop_1=model_lstm_c.fit(train_x,comment_train_target,epochs=10,batch_size=128,validation_split=0.1)

#0.3
inp = Input(shape=(maxlen,))
x = Embedding(max_f,embed_size)(inp)
x = SpatialDropout1D(0.35)(x)
x = Conv1D(64, kernel_size=3, padding='valid', kernel_initializer='glorot_uniform')(x)
x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(64, activation="relu")(x)
x= Dropout(0.3)(x)
x= Dense(6,activation='sigmoid') (x)
model_lstm_c = Model(inputs=inp, outputs=x)
model_lstm_c.compile(loss='binary_crossentropy',optimizer= 'adam',metrics=['accuracy'])
drop_3=model_lstm_c.fit(train_x,comment_train_target,epochs=10,batch_size=128,validation_split=0.1)

#0.5
inp = Input(shape=(maxlen,))
x = Embedding(max_f,embed_size)(inp)
x = SpatialDropout1D(0.35)(x)
x = Conv1D(64, kernel_size=3, padding='valid', kernel_initializer='glorot_uniform')(x)
x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(64, activation="relu")(x)
x= Dropout(0.5)(x)
x= Dense(6,activation='sigmoid') (x)
model_lstm_c = Model(inputs=inp, outputs=x)
model_lstm_c.compile(loss='binary_crossentropy',optimizer= 'adam',metrics=['accuracy'])
drop_5=model_lstm_c.fit(train_x,comment_train_target,epochs=10,batch_size=128,validation_split=0.1)

#0.7
inp = Input(shape=(maxlen,))
x = Embedding(max_f,embed_size)(inp)
x = SpatialDropout1D(0.35)(x)
x = Conv1D(64, kernel_size=3, padding='valid', kernel_initializer='glorot_uniform')(x)
x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(64, activation="relu")(x)
x= Dropout(0.7)(x)
x= Dense(6,activation='sigmoid') (x)
model_lstm_c = Model(inputs=inp, outputs=x)
model_lstm_c.compile(loss='binary_crossentropy',optimizer= 'adam',metrics=['accuracy'])
drop_7=model_lstm_c.fit(train_x,comment_train_target,epochs=10,batch_size=128,validation_split=0.1)

#plot for drop out rate
plt.suptitle('drop out rate difference', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.plot(drop_1.history['val_loss'], color='b', label='Validation Loss_0.1')
plt.plot(drop_3.history['val_loss'], color='r', label='Validation Loss_0.3')
plt.plot(drop_5.history['val_loss'], color='g', label='Validation Loss_0.5')
plt.plot(drop_7.history['val_loss'], color='y', label='Validation Loss_0.7')
plt.legend(loc='upper right')


#final model 
inp = Input(shape=(maxlen,))
x = Embedding(max_f,embed_size)(inp)
x = SpatialDropout1D(0.35)(x)
x = Conv1D(64, kernel_size=3, padding='valid', kernel_initializer='glorot_uniform')(x)
x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(64, activation="relu")(x)
x= Dropout(0.1)(x)
x= Dense(6,activation='sigmoid') (x)
model_lstm_c = Model(inputs=inp, outputs=x)
model_lstm_c.compile(loss='binary_crossentropy',optimizer= 'adam',metrics=['accuracy'])
final_model=model_lstm_c.fit(train_x,comment_train_target,epochs=10,batch_size=128,validation_split=0.1)

#plot for final model loss
plt.suptitle('Final model', fontsize=10)
plt.ylabel('loss', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.plot(final_model.history['val_loss'], color='b', label='Validation Loss')
plt.plot(final_model.history['loss'], color='r', label='Loss')
plt.legend(loc='upper right')

#plot for final model accuracy
plt.suptitle('Final model', fontsize=10)
plt.ylabel('accuracy', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.plot(final_model.history['accuracy'], color='g', label='accuracy')
plt.plot(final_model.history['val_accuracy'], color='y', label='val_accuracy')
plt.legend(loc='upper right')

#final submission
y_pred_con_l=model_lstm_c.predict(submmit_x,batch_size=128)
submid = pd.DataFrame({'id': comment_test["id"]})
submission = pd.concat([submid, pd.DataFrame(y_pred_con_l, columns = cat)], axis=1)
submission[["toxic","severe_toxic","obscene","threat","insult","identity_hate"]]=y_pred_con_l
submission.to_csv('submission.csv',index=False)