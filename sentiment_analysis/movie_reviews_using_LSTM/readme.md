# What is Sequential problems?
Sequential problems are a class of problem in machine learning in which the
order of the features presented to the model is important for making predictions

# When these problem usually arise?
Sequential problems are commonly encountered in the following
scenarios:
1. NLP, including sentiment analysis, language translation, and text prediction
2. Time series predictions

# What is RNN(Recurrent neural networks)?
![RNN](images/rnn.png)

RNN is a multi-layered neural network. If the raw input is a sentence, we can break up the sentence into
individual words (in this case, every word represents a time step). Each word
will then be provided in the corresponding layer in the RNN as Input. More
importantly, each layer in an RNN passes its output to the next layer. The
intermediate output passed from layer to layer is known as the hidden state.
Essentially, the hidden state allows an RNN to maintain a memory of the
intermediate states from the sequential data.

# What's inside an RNN?
![Inside of RNN](images/What's_inside_an_RNN.png)

The mathematical function of an RNN is simple. Each layer t within an RNN has two inputs:
The input from the time step t
The hidden state passed from the previous layer t-1
Each layer in an RNN simply sums up the two inputs and applies a tanh
function to the sum. It then outputs the result, to be passed as a hidden state to
the next layer. It's that simple! More formally, the output hidden state of layer
t is this:
![Activation Functions](images/activation_functions.png)

# What exactly is the tanh function? 
The tanh function is the hyperbolic tangent function, and it simply squashes a value between 1 and -1. 
The tanh function is a good choice as a non-linear transformation of the combination of the current input 
and the previous hidden state, because it ensures that the weights don't diverge too rapidly. It has also 
other nice mathematical properties, such as being easily differentiable.

# What is LSTM(long short-term memory)?
Learn lstm by example

We can treat this short sentence as sequential data by breaking it down into five different inputs, with each word at each time step
![LSTM](images/lstm_1.png)
Now, suppose that we are building a simple RNN to predict whether is it snowing based on this sequential data
![LSTM](images/lstm_2.png)

The critical piece of information in the sequence is the word HOT, at time step 4 (t4, circled in red). With this piece of information, 
the RNN is able to easily predict that it is not snowing today. Notice that the critical piece of information came just shortly before 
the final output. In other words, we would say that there is a short-term dependency in this sequence.

> # Long Term memory with a long sentence example

![LTM](images/LT_1.png)
> Our goal is to predict whether the customer liked the movie.

Clearly, the customer liked the movie but not the cinema, which was the main complaint in the paragraph.

![LTM](images/LT_2.png)