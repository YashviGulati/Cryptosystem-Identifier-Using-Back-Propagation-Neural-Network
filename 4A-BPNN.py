
# coding: utf-8

# # Crytosystem Identifier Using Back Propagation Neural Network

# In[34]:


import numpy as np
import os
from sklearn.utils import shuffle
from sklearn.feature_extraction import image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
import keras
import sklearn
from keras.models import Sequential
from keras.layers import Dense
import collections
from collections import Counter


# In[35]:


#Function for Reading Data from a Text File
def read_text_file(fname):
    with open(fname,'r') as f:
        text=f.readlines()
    for i in range(len(text)):
        text[i]=text[i].rstrip("\n")
    return text

#Function for Converting the text array into its respective ASCII value.
def txt_char(arr,size=200):
    a=len(arr)
    res=32*np.ones((a,size))
    for i in range(a):
        arr[i]+=" "
        for j in range(size):
            res[i,j]=ord(arr[i][j])
    return res


# In[36]:


#We have Extracted frequency features from the Data.
def extract_features(arr,f=7):
    res=[]
    for i in arr:
        a=Counter(i)
        r=list(a.values())
        s=np.sort(r)
        res.append([np.mean(s[:f]),np.mean(s[-1*f:])])
    return np.array(res)


# In[37]:


def train(train_features,train_labels,valid_features,valid_labels,hid_1,input_dim=2,epochs=50,verb=2):
    classifier = Sequential()

    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units = hid_1, kernel_initializer = 'uniform', activation = 'relu', input_dim = 2))

    # Adding the output layer
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    # Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Fitting the ANN to the Training set
    classifier.fit(train_features, train_labels, batch_size = 10, epochs = 50, validation_data=(valid_features,valid_labels),verbose=verb)
    
    return classifier


# In[38]:


#Reading of Training Data
sub_data=read_text_file(os.path.join('Dataset','sub_train.txt')) 
vig_data=read_text_file(os.path.join('Dataset','vig_train.txt')) 

shuffle(sub_data)
shuffle(vig_data)

sub_train,sub_val=sub_data[:40],sub_data[40:]
vig_train,vig_val=vig_data[:40],vig_data[40:]

#Reading of Testing Data
sub_test=read_text_file(os.path.join('Dataset','sub_test.txt'))
vig_test=read_text_file(os.path.join('Dataset','vig_test.txt'))

sen_train=np.asarray(sub_train+vig_train)
sen_valid=np.asarray(sub_val+vig_val)
sen_test=np.asarray(sub_test+vig_test)

train_data=txt_char(sen_train)
valid_data=txt_char(sen_valid)
test_data=txt_char(sen_test)

train_labels=np.asarray([0]*len(sub_train)+[1]*len(vig_train))
valid_labels=np.asarray([0]*len(sub_val)+[1]*len(vig_val))
test_labels=np.asarray([0]*len(sub_test)+[1]*len(vig_test))


# In[39]:


freq=5   ### FREQUENCY VECTOR DECIDER
neuron_variation=list(range(5,101,5)) # No. of neurons varys from 5 to 100 in step of 5
Accuracy_neuron=[] 
for hid1 in neuron_variation:
    train_features=extract_features(train_data,freq)
    valid_features=extract_features(valid_data,freq)
    test_features=extract_features(test_data,freq)

    train_labels=train_labels.reshape((train_labels.shape[0],1))
    valid_labels=valid_labels.reshape((valid_labels.shape[0],1))
    test_labels=test_labels.reshape((test_labels.shape[0],1))
    classifier=train(train_features,train_labels,valid_features,valid_labels,hid1,verb=1,epochs=100)
    test_out=classifier.predict(test_features)
    test_predict=test_out>0.5
    accuracy=(test_predict==test_labels).sum()/20.0
    print (hid1,accuracy)
    Accuracy_neuron.append(accuracy)
    print(" ")


# In[40]:


print(np.vstack([neuron_variation, Accuracy_neuron]).T)


# In[41]:


hid_neuron=50 ## No. of neurons in hidden layer
frequency_variation=list(range(1,9,1)) ## Frequency decider varies from 1 to 8
Accuracy_freq=[]
for fq in frequency_variation:
    train_features=extract_features(train_data,fq)
    valid_features=extract_features(valid_data,fq)
    test_features=extract_features(test_data,fq)

    train_labels=train_labels.reshape((train_labels.shape[0],1))
    valid_labels=valid_labels.reshape((valid_labels.shape[0],1))
    test_labels=test_labels.reshape((test_labels.shape[0],1))
    classifier=train(train_features,train_labels,valid_features,valid_labels,hid_neuron,verb=0,epochs=100)
    test_out=classifier.predict(test_features)
    test_predict=test_out>0.5
    accuracy=(test_predict==test_labels).sum()/20.0
    print(fq,accuracy)
    Accuracy_freq.append(accuracy)


# In[42]:


print(np.vstack([frequency_variation, Accuracy_freq]).T)


# ## After this use matplotlib to plot accuracy vs frequency curve and Accuracy vs neuron_variation curve

# In[43]:


plt.plot(frequency_variation,Accuracy_freq)
plt.xlabel('Frequency')
plt.ylabel('Accuracy')
plt.title('Frequency VS Accuracy')


# In[44]:


plt.plot(neuron_variation, Accuracy_neuron)
plt.xlabel('Number of Neurons')
plt.ylabel('Accuracy')
plt.title('Number of Neurons VS Accuracy')

