# Folder Structure

```bash
.
├── attention
│   ├── compare.py
│   ├── connectivity_visualisation.py
│   ├── evaluation.py
│   ├── heatmaps.py
│   └── sweep.py
├── best_attention_model.pt
├── best_vanilla_model.pt
├── comparison.csv
├── dakshina_dataset_v1.0
│   └── hi
│       └── lexicons
│           ├── hi.translit.sampled.dev.tsv
│           ├── hi.translit.sampled.test.tsv
│           └── hi.translit.sampled.train.tsv
├── font
│   └── NotoSansDevanagari-Regular.ttf
├── main.ipynb
├── model.py
├── predictions_attention
│   └── predictions_attention.csv
├── predictions_vanilla
│		└── predictions_vanilla.csv
├── requirements.txt
├── sentence_attention.html
├── tree.txt
├── utils
│   ├── helper_functions.py
│   ├── model_evaluation.py
│   └── sweeper.py
└── vanilla
    ├── evaluation.py
    └── sweep.py
```

# Setup Instructions:

- Clone the repository and move to part A code
    
    ```
    git clone https://github.com/Harsh-trivs/da6401_assignment3.git
    cd da6401_assignment3.git
    ```
    
- Create a virtual environment
    
    ```
    python -m venv venv
    ```
    
    Activate virtual environment
    
    ```
    .\venv\Scripts\activate # Windows
    source venv/bin/activate # Mac/Linux
    ```
    
- Install dependencies
    
    ```
    pip install -r requirements.txt
    ```
    

# Utils:

## Sweeper.py:

Used for running wandb sweeps provided with the config and training function. Logging accuracy and loss on validation as well as train data set.

## Eval.py :

Used for generating the evaluation of the model on the test dataset logging test accuracy and generating prediction tables.

# Vanilla model (Python files used for experiments)

## Sweep.py :

Calls helper function for running Weights & Biases (wandb) sweeps to perform hyperparameter tuning on the custom-built model.

## Evaluation.py :

Calls helper function to evaluate the best vanilla model on test data. Logging test accuracy and a table showing predictions on test data.

# Attention model (Python files used for experiments)

## Sweep.py :

Calls helper function for running Weights & Biases (wandb) sweeps to perform hyperparameter tuning on the custom-built model.

## Evaluation.py :

Calls helper function to evaluate the best vanilla model on test data. Logging test accuracy and a table showing predictions on test data.

## Compare.py :

Compare models with and without attention. The logging table contains elements predicted correctly by the attention model but incorrectly by the vanilla model.

## Heatmaps.py:

Use model attention weights to generate attention heatmaps for the attention-based model.

## Connectivity_visualisation.py :

Function to generate connectivity visualization on samples from the test set using the best model and logging HTML generated to wandb.

# GitHub Link :

https://github.com/Harsh-trivs/da6401_assignment3.git

# Wandb Link :

https://wandb.ai/harshtrivs-indian-institute-of-technology-madras/dakshina_transliteration/reports/Assignment-3--VmlldzoxMjgzOTY0Ng?accessToken=vp3yeafg77ma9qpujb7fnd7ugvpp2m78hzqsnhr7xuyp1tja38c5dncr3l5wlsge
