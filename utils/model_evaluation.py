import os
import random
from model import TransliterationModel
import torch
import pandas as pd
from utils.helper_functions import build_vocab_and_prepare_batch, compute_word_level_accuracy, read_pairs
import wandb

import torch.nn as nn
import torch.optim as optim

def model_eval(cfg,model_path,project_name,csv_file_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wandb.finish()
    wandb.init(
        project=project_name,
        name = 'best_model_test_eval',
        resume="never",
        reinit=True,
        config=cfg
    )
    # Load and prepare data
    train_path = "dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv"
    train_set = read_pairs(train_path)
    test_path = "dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv"
    test_set = read_pairs(test_path)

    src_vocab, idx2src, tgt_vocab, idx2tgt, create_batch, _, _ = build_vocab_and_prepare_batch(train_set, device)

    # Initialize model, optimizer, criterion
    model = TransliterationModel(
        len(src_vocab), len(tgt_vocab), cfg['embedding_size'], cfg['hidden_size'],
        cfg['enc_layers'], cfg['dec_layers'], cfg['rnn_type'], cfg['dropout_rate'],
        cfg['is_bidirectional'],cfg['use_attention']
    ).to(device)
    if not os.path.exists(model_path):
        print("❌ No saved model found, starting training.")
        optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])
        criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<pad>'])
        best_acc = 0.0
        # Training loop
        for epoch in range(cfg['epochs']):
            model.train()
            total_loss, total_acc = 0, 0
            random.shuffle(train_set)

            for i in range(0, len(train_set), cfg['batch_size']):
                batch = train_set[i:i+cfg['batch_size']]
                src, tgt = create_batch(batch)

                optimizer.zero_grad()
                if cfg['use_attention']:
                    outputs, attn_weights = model(src, tgt, cfg['teacher_forcing_prob'])
                else:
                    outputs = model(src, tgt, cfg['teacher_forcing_prob'])

                loss = criterion(outputs[:, 1:].reshape(-1, outputs.size(-1)), tgt[:, 1:].reshape(-1))
                loss.backward()
                optimizer.step()

                preds = outputs.argmax(-1)
                acc = compute_word_level_accuracy(preds[:, 1:], tgt[:, 1:], tgt_vocab)

                total_loss += loss.item()
                total_acc += acc

            avg_train_loss = total_loss / (len(train_set) // cfg['batch_size'])
            avg_train_acc = total_acc / (len(train_set) // cfg['batch_size'])

            print(f"Epoch {epoch+1}/{cfg['epochs']} | Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.2f}%")
            wandb.log({"Train Loss": avg_train_loss, "Train Accuracy": avg_train_acc})

            # Save the best model
            if avg_train_acc > best_acc:
                best_acc = avg_train_acc
                torch.save(model.state_dict(), model_path)
                print(f"💾 Saved new best model at epoch {epoch + 1} with accuracy {best_acc:.2f}%")
        print(f"Best model saved with accuracy: {best_acc:.2f}%")

    #test the model
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("✅ Loaded saved model from disk.")
    model.eval()
    predictions = []

    with torch.no_grad():
        for i in range(0, len(test_set), cfg['batch_size']):
            batch = test_set[i:i + cfg['batch_size']]
            src, tgt = create_batch(batch)
            if cfg['use_attention']:
                outputs, attn_weights = model(src, tgt, 0)
            else:
                outputs = model(src, tgt, 0)
            preds = outputs.argmax(-1)

            for j in range(src.size(0)):
                input_seq = ''.join([idx2src.get(idx.item(), '') for idx in src[j] if idx.item() not in [src_vocab['<pad>'], src_vocab['<eos>']]])
                target_seq = ''.join([idx2tgt.get(idx.item(), '') for idx in tgt[j][1:] if idx.item() not in [tgt_vocab['<pad>'], tgt_vocab['<eos>']]])
                pred_seq = ''.join([idx2tgt.get(idx.item(), '') for idx in preds[j][1:] if idx.item() not in [tgt_vocab['<pad>'], tgt_vocab['<eos>']]])
                is_correct = target_seq == pred_seq
                predictions.append({'Input': input_seq, 'Target': target_seq, 'Predicted': pred_seq , 'Is_Correct': "True✅" if is_correct else "False❌"})
    predictions = pd.DataFrame(predictions)
    overall_acc = (predictions.Is_Correct == "True✅").mean()
    wandb.log({"Test Accuracy": overall_acc})
    table = wandb.Table(dataframe=predictions)
    wandb.log({f"{csv_file_name}_table": table})
    # finish run
    wandb.finish()
    predictions.to_csv(f'{csv_file_name}.csv', index=False)
    print(f"Saved {len(predictions)} rows, eval accuracy = {overall_acc:.2f}")
