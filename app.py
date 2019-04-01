#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

# # Pre-processing data
# * 第一步簡單來說就是看一下Data是什麼樣子
# * 第二步就是取前60天的MW作爲training的input，後7天作爲output
# * 取出2019年4月1日以及前60天，作爲真正要預測的input
#
# # Data Normallize
# * 就簡單的減去最小值，再除以最大值減去最小值的差就完事了

# In[2]:


data = pd.read_csv('data.csv')
print(data.head())  # check data

# In[3]:


# 構造trainning data
target = data['尖峰負載(MW)']
date = data['日期']

train_X = []
train_Y = []

mx = np.max(target)
mn = np.min(target)
for i in range(len(target) - 67):
    temp_X = (target[i: i + 60] - mn) / (mx - mn + 233)
    temp_Y = (target[i + 60: i + 60 + 7] - mn) / (mx - mn + 233)

    # print(temp_Y.values)
    train_X.append(np.reshape(np.array(temp_X.values), (60, 1)))
    train_Y.append(np.array(temp_Y.values))

submit_X = np.reshape(np.array((target[-60:, ] - mn) / (mx - mn + 233)), (1, 60, 1))

# In[4]:


# 構造predict的input
submit_X = np.reshape(np.array((target[-60:, ] - mn) / (mx - mn + 233)), (1, 60, 1))

# In[5]:


# check一下shape ， 看看有多少資料， 發現少的可憐QAQ
train_X = np.array(train_X)
train_Y = np.array(train_Y)

print(train_X.shape)
print(train_Y.shape)

# # 構造CNN模型
# * 對於sequencial的問題，大家一般會想到用RNN來處理，但是我只有700多筆資料少的可憐，RNN基本上是學不到什麼東西的QQ
# * 通過觀察可以發現MW是一個有着非常非常明顯週期性的東西，爾一維的CNN，可以理解爲差分操作，剛好可以查找出明顯週期性的特徵
# * 然後就是用CNN寫一個簡單的model，加點dropout可以解決過擬合
# * 資料太少，做3Fold取平均

# In[6]:


import keras as K


def build_rnn():
    input_x = K.layers.Input(shape=[60, 1])
    X = K.layers.Conv1D(30, 10, strides=1, padding='valid')(input_x)
    X = K.layers.Flatten()(X)
    X = K.layers.Dense(60, activation='relu', name='dense0')(X)
    X = K.layers.Dropout(0.5)(X)
    result = K.layers.Dense(7, activation='sigmoid', name='output')(X)
    model = K.Model(input=input_x, output=result)

    return model


# In[7]:


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse
from keras import callbacks as kc

submit = np.zeros(shape=(1, 7), dtype=np.float32);

kfold = KFold(n_splits=3, shuffle=True, random_state=3)
opt = K.optimizers.Adam(lr=0.001)
scores = []
for train_index, valid_index in kfold.split(train_X, train_Y):
    X_tr = train_X[train_index]
    Y_tr = train_Y[train_index]
    X_val = train_X[valid_index]
    Y_val = train_Y[valid_index]
    callbacks = [kc.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)]
    rnn = build_rnn()
    rnn.compile(optimizer=opt, loss='mean_squared_error')
    rnn.fit(X_tr, Y_tr, batch_size=200, epochs=500, validation_data=[X_val, Y_val], callbacks=callbacks, verbose=0)
    pre_val = rnn.predict(X_val)
    score = mse(pre_val, Y_val)
    print(Y_val[0], pre_val[0])
    scores.append(score)
    result = rnn.predict(submit_X)[0]
    submit += result;
    print(result)

# # 計算預期得分
# 由期望的性質可以知道，預計得分就是sqrt(CV score * (maxvalue - minvalue)^2)

# In[8]:


print(scores)
import math

mean = np.mean(scores) * (mx - mn + 233) * (mx - mn + 233)
std = np.std(scores) * (mx - mn + 233) * (mx - mn + 233)

print("預測分數應該在", math.sqrt(mean), "+-", math.sqrt(std))

# In[9]:


submit /= 3
submit = mn + submit * (mx - mn + 233)

# In[10]:


print(submit)

# In[11]:


col = ['date', 'peak_load(MW)']
submission = pd.DataFrame()
submission['date'] = [date for date in range(20190402, 20190408 + 1)]
submission['peak_load(MW)'] = submit[0];

# In[12]:


print(submission)

# # 面向節日編程
# * 剛好遇到清明節是真的難受，所以我把清明放假的結果替換爲前兩年的清明假期的均值QAQ
# * 同時爲了尊重一下我的model的結果， 我用98比2的比例，再和我的結果加權平均。

# In[13]:


festival_2017 = np.array([24245, 22905, 22797, 23637])
festival_2018 = np.array([24981, 24450, 23940, 23895])

replace = (festival_2017 + festival_2018) / 2

for i in range(4):
    submission['peak_load(MW)'][2 + i] = submission['peak_load(MW)'][2 + i] * 0.02 + replace[i] * 0.98

# In[14]:


print(submission)

# In[17]:


submission.to_csv('submission.csv', index=0)
