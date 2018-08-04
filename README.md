

------------------
Problem Statement:
------------------
Cryptosystem Identifier Using Back Propagation Neural Network. It classifies the text into either Vigenere or Simple Substitution. Return '0' if it is Simple Substitution Cipher and Return '1' if it is Vigenere Cipher.

-----------
Libraries
-----------
numpy,os,sklearn,matplotlib,keras (tensorflow at backend),collections

----------
Functions
----------

------------------------
a. read_text_file(fname)
------------------------
Takes an argument fname i.e. the path of the file that has to be read, reads the text file and returns an array of characters.

---------------------
b. txt_char(arr,size)
---------------------
Takes two arguements i.e. array of characters and the size of the cryptogram (size (number of characters in each cryptogram=200). It generates the ASCII value for each character and returns an array of ASCII values.

--------------------------
c. extract_features(arr,f)
--------------------------
Takes two arguements i.e. array of ASCII values and number of elements taken for average. It extracts the basic features and returns a feature array.

----------------------------------------------------------------------------------------------------
d. train(train_features,train_labels,valid_features,valid_labels,hid_1,input_dim=2,epochs=50,verb=2)
----------------------------------------------------------------------------------------------------
It takes following arguements:
=> train_features: It is the feature array which goes into the first layer(input layer)
=>train_labels: It is the true value or the ground truth of all the training dataset.
=>valid_features: These are features extracted from the validation data.
=>valid_labels: It is the true value or the ground truth of all the validation dataset
=>hid_1: It determines the total number of neurons in hidden layer
=>input_dim: It defines the shape of the input value.
=>epochs: Total number of times, the complete code should run.

The main model implementation is done here. Activation Functions, RelU and Sigmoid are used.

Layer (type)                 Output Shape              Param #  
=================================================================
dense_167 (Dense)            (None, 50)                150      
_________________________________________________________________
dense_168 (Dense)            (None, 1)                 51        
=================================================================
Total params: 201
Trainable params: 201
Non-trainable params: 0
_________________________________________________________________









 
