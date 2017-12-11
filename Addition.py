import numpy as np
import tensorflow as tf
from numpy import sum
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf


#Defining some hyper-params
num_units = 50
input_size = 1
batch_size = 50
seq_len = 3
drop_out = 0.8


#Creates our random sequences
def gen_data(min_length=3, max_length=15, n_batch=50):
    X = np.concatenate([np.random.randint(10,size=(n_batch, max_length, 1))],axis=-1)
    y = np.zeros((n_batch,))
    # Compute masks and correct values
    for n in range(n_batch):
        length = seq_len
        X[n, length:, 0] = 0
        y[n] = np.sum(X[n, :, 0]*1)
    return (X,y)

### Model Construction
num_layers = 1
encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
cell = tf.nn.rnn_cell.MultiRNNCell([encoder_cell] * num_layers)
cell = tf.nn.rnn_cell.DropoutWrapper(cell,output_keep_prob=drop_out)

inputs = [tf.placeholder(tf.float32,shape=[batch_size,1]) for _ in range(seq_len)]
result = tf.placeholder(tf.float32, shape=[batch_size])
initial_state = cell.zero_state(batch_size, tf.float32)
print(tf.shape(initial_state))
print(tf.shape(inputs))
print()

outputs, states = tf.contrib.legacy_seq2seq.rnn_decoder(inputs, initial_state, cell, scope ='rnn')
outputs2 = outputs[-1]

W_o = tf.Variable(tf.random_normal([num_units,input_size], stddev=0.01))
b_o = tf.Variable(tf.random_normal([input_size], stddev=0.01))

outputs3 = tf.matmul(outputs2, W_o) + b_o

cost = tf.pow(tf.subtract(tf.reshape(outputs3, [-1]), result),2)

train_op = tf.train.RMSPropOptimizer(0.005, 0.2).minimize(cost)


### Generate Validation Data
tempX,y_val = gen_data(2,seq_len,batch_size)
X_val = []
for i in range(seq_len):
    X_val.append(tempX[:,i,:])

##Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
train_score =[]
val_score= []
x_axis=[]

num_epochs=1000
for k in range(1,num_epochs):

    #Generate Data for each epoch
    tempX,y = gen_data(2,seq_len,batch_size)
    X = []
    for i in range(seq_len):
        X.append(tempX[:,i,:])

    temp_dict = {inputs[i]:X[i] for i in range(seq_len)}
    temp_dict.update({result: y})

    _,c_train = sess.run([train_op,cost],feed_dict=temp_dict)   #perform an update on the parameters

    val_dict = {inputs[i]:X_val[i] for i in range(seq_len)}  #create validation dictionary
    val_dict.update({result: y_val})
    c_val = sess.run([cost],feed_dict = val_dict )            #compute the cost on the validation set
    if (k%100==0):
        train_score.append(sum(c_train))
        val_score.append(sum(c_val))
        x_axis.append(k)

print("Final Train cost: {}, on Epoch {}".format(train_score[-1],k))
print("Final Validation cost: {}, on Epoch {}".format(val_score[-1],k))

val_score_v =[]
num_epochs=1

for k in range(num_epochs):
    tempX,y = gen_data(2,seq_len,batch_size)
    X = []
    for i in range(seq_len):
        X.append(tempX[:,i,:])

    val_dict = {inputs[i]:X[i] for i in range(seq_len)}
    val_dict.update({result: y})
    outv, c_val = sess.run([outputs3,cost],feed_dict = val_dict ) 
    val_score_v.append([c_val])
print("Validation cost: {}, on Epoch {}".format(c_val,k))

for i in range(batch_size):
    print(tempX[i])
    print("expected", y[i])
    x =  round(outv[i][-1],2)
    print("actual", x)
    print("difference :", abs(y[i] - x))
    print("\n")