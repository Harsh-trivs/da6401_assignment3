import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.model_evaluation import model_eval


parameters = {
        'embedding_size':256,
        'hidden_size': 256,
        'enc_layers': 2,
        'dec_layers': 3,
        'rnn_type': 'LSTM',
        'dropout_rate': 0.3,
        'batch_size': 64,
        'epochs':10,
        'is_bidirectional':True,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'teacher_forcing_prob':0.7,
        'beam_size': 3,
        'use_attention': True,
    }
model_eval(parameters,"best_attention_model.pt","transliteration_attention_evaluation","predictions_attention")