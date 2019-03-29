# Neural_Network_Dependency_Parsing

This model is to train a feed-forward neural network to predict the transitions of an
arc-standard dependency parser. The input to this network will be a representation of the current
state (including words on the stack and buffer). The output will be a transition (shift, left_arc,
right_arc), together with a dependency relation label

I use the Keras package to construct the neural net. Keras is a high-level Python API that
allows you to easily construct, train, and apply neural networks. However, Keras is not a neural
network library itself and depends on one of several neural network backends. I will use the
Tensorflow backend.