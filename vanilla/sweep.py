import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.sweeper import run_training
import wandb

sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'Validation Accuracy', 'goal': 'maximize'},
    'parameters': {
        'embedding_size': {'values': [128, 256]},
        'hidden_size': {'values': [128, 256]},
        'enc_layers': {'values': [2, 3]},
        'dec_layers': {'values': [2, 3]},
        'rnn_type': {'values': ['GRU', 'LSTM','RNN']},
        'dropout_rate': {'values': [0.2, 0.3]},
        'batch_size': {'values': [32, 64]},
        'epochs': {
            'values': [5, 10]},
        'is_bidirectional': {'values': [False, True]},
        'learning_rate': {'values': [0.001, 0.0001]},
        'optimizer': {'values': ['adam', 'nadam']},
        'teacher_forcing_prob': {'values': [0.2, 0.5, 0.7]},
        'beam_size': {'values': [1,3,5]},
        'use_attention': {'values': [False]},
    }
}

sweep_id = wandb.sweep(sweep_config, project="dakshina_transliteration")
wandb.agent(sweep_id, function=run_training, count=50)
