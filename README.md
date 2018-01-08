# Recurrent-Neural-Network-Sum-of-digits-in-a-number
Addition of Two numbers using RNN Tensorflow

This model is inspired by a blog post on RNN by Rajiv Shash (http://projects.rajivshah.com/blog/2016/04/05/rnn_addition/)
The code uses Python 3.6 to run along with Tensorflow.
The code generates the dataset randomly using random.randint() of python and generates 3 digit numbers as datasets and calculates the lables as their sum for training.
The model inputs the generates data in batches of 50, as a sequence in the order of a digit after digit. The algorithm trains the model using seq2seq method of tensorflow and outputs the sum wich is more or less near the expected answer.
