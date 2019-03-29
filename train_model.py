from extract_training_data import FeatureExtractor
import sys
import numpy as np
import keras
from keras import Sequential
from keras.layers import Flatten, Embedding, Dense

def build_model(word_types, pos_types, outputs):
    # TODO: Write this function for part 3
    # Using the Keras package to build the neural network
    model = Sequential()

    # One Embedding layer, the input_dimension should be the number possible words, the
    # input_length is the number of words using this same embedding layer. This should be 6,
    # because we use the 3 top-word on the stack and the 3 next words on the buffer.
    # The output_dim of the embedding layer should be 32.
    model.add(Embedding(word_types, 32, input_length=6))

    # to Flatten the output of the embedding layer first
    model.add(Flatten())

    # A Dense hidden layer of 100 units using relu activation.
    model.add(Dense(100, activation='relu'))

    # A Dense hidden layer of 10 units using relu activation
    model.add(Dense(10, activation='relu'))

    # An output layer using softmax activation
    model.add(Dense(91, activation='softmax'))

    # The method should prepare the model for training, in this case using categorical
    # crossentropy as the loss and the Adam optimizer with a learning rate of 0.01
    model.compile(keras.optimizers.Adam(lr=0.01), loss="categorical_crossentropy")
    return model


if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    print("Compiling model.")
    model = build_model(len(extractor.word_vocab), len(extractor.pos_vocab), len(extractor.output_labels))
    inputs = np.load(sys.argv[1])
    outputs = np.load(sys.argv[2])
    print("Done loading data.")

    # Now train the model
    model.fit(inputs, outputs, epochs=5, batch_size=100)

    model.save(sys.argv[3])

    # extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    # print("Compiling model.")
    # model = build_model(len(extractor.word_vocab), len(extractor.pos_vocab), len(extractor.output_labels))
    # inputs = np.load('data/input_train.npy ')
    # outputs = np.load('data/target_train.npy')
    # print("Done loading data.")
    #
    # # Now train the model
    # model.fit(inputs, outputs, epochs=5, batch_size=100)
    #
    # model.save('data/model.h5')
