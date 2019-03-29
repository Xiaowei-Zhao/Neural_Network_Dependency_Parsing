from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import keras

from extract_training_data import FeatureExtractor, State

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        
        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        # The function first creates a State instance in the initial state, i.e. only word 0 is on the stack, the
        # buffer contains all input words (or rather, their indices) and the deps structure is empty
        state = State(range(1,len(words)))
        state.stack.append(0)    

        # The algorithm is the standard transition-based algorithm. As long as the buffer is not empty, we
        # use the feature extractor to obtain a representation of the current state
        while state.buffer:
            # TODO: Write the body of this loop for part 4
            transitions = []
            if not state.stack:
                transitions.append('shift')
            elif state.stack[-1] == 0:
                transitions.append('right_arc')
                if len(state.buffer) > 1:
                    transitions.append('shift')
            else:
                transitions.append('right_arc')
                transitions.append('left_arc')
                if len(state.buffer) > 1:
                    transitions.append('shift')
            #print(transitions)

            # As long as the buffer is not empty,
            # use the feature extractor to obtain a representation of the current state
            input_rep = self.extractor.get_input_representation(words, pos, state).reshape((1, 6))
            # print(input_rep)

            # Call model.predict(features) and retrieve a softmax actived vector of possible actions
            predict = self.model.predict(input_rep)[0]
            #print(predict)
            index = list(np.argsort(predict)[::-1])
            #print(index)

            for i in index:
                action, label = self.output_labels[i]
                if action in transitions:
                    if action == 'shift':
                        state.shift()
                    elif action == 'left_arc':
                        state.left_arc(label)
                    else:
                        state.right_arc(label)
                    break

        result = DependencyStructure()
        for p,c,r in state.deps: 
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result 
        

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
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2], 'r') as in_file:
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()

    # extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    # parser = Parser(extractor, 'data/model.h5')
    #
    # with open('data/dev.conll', 'r') as in_file:
    #     for dtree in conll_reader(in_file):
    #         words = dtree.words()
    #         pos = dtree.pos()
    #         deps = parser.parse_sentence(words, pos)
    #         print(deps.print_conll())
    #         print()
        
