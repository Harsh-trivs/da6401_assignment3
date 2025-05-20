import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.model_evaluation import model_eval

#best vanilla model parameters
parameters = {
        'embedding_size':256,
        'hidden_size': 256,
        'enc_layers': 3,
        'dec_layers': 3,
        'rnn_type': 'GRU',
        'dropout_rate': 0.3,
        'batch_size': 64,
        'epochs':10,
        'is_bidirectional':False,
        'learning_rate': 0.001,
        'optimizer': 'nadam',
        'teacher_forcing_prob':0.7,
        'beam_size': 5,
        'use_attention': False,
    }
model_eval(parameters,"best_vanilla_model.pt","transliteration_evaluation","predictions_vanilla")