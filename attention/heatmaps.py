import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
from io import BytesIO
from PIL import Image
from model import TransliterationModel
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from utils.helper_functions import build_vocab_and_prepare_batch, compute_word_level_accuracy, read_pairs
import wandb

# Assumes TransliterationModel, build_vocab_and_prepare_batch, compute_word_level_accuracy, and read_pairs are defined elsewhere

def plot_attention(attn_weights, input_tokens, output_tokens, input_word, output_word, idx):
    # Load Devanagari-compatible font
    font_path = r"font\NotoSansDevanagari-Regular.ttf"
    devanagari_font = fm.FontProperties(fname=font_path, size=12)  # Increase font size here

    # Define special tokens to exclude
    special_tokens = {'<pad>', '<sos>', '<eos>'}
    filtered_input_tokens = [tok for tok in input_tokens if tok not in special_tokens]
    filtered_output_tokens = [tok for tok in output_tokens if tok not in special_tokens]
    attn_weights_filtered = attn_weights[:len(filtered_output_tokens), :len(filtered_input_tokens)]

    # Create heatmap
    plt.figure(figsize=(8, 6))  # Slightly larger figure
    ax = sns.heatmap(attn_weights_filtered,
                     xticklabels=filtered_input_tokens,
                     yticklabels=filtered_output_tokens,
                     cmap='viridis',
                     cbar_kws={"shrink": 0.7})

    # Set font sizes for labels and title
    ax.set_xlabel("Input Sequence (characters)", fontsize=16)
    ax.set_ylabel("Predicted Output (characters)", fontproperties=devanagari_font, fontsize=16)
    ax.set_title(f"Heatmap {idx}: '{input_word}' - '{output_word}'", fontproperties=devanagari_font, fontsize=16)

    # Set tick label font sizes
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    # Apply Devanagari font to y-tick labels
    for label in ax.get_yticklabels():
        label.set_fontproperties(devanagari_font)

    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    plt.tight_layout()

    # Convert to wandb.Image
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    image = Image.open(buf)

    return wandb.Image(image, caption=f"{idx}: {input_word} → {output_word}")



def attention_heatmaps(cfg, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wandb.finish()
    wandb.init(
        project="transliteration_attention_heatmap",
        name='best_attention_model_test_eval',
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

    # Initialize model
    model = TransliterationModel(
        len(src_vocab), len(tgt_vocab), cfg['embedding_size'], cfg['hidden_size'],
        cfg['enc_layers'], cfg['dec_layers'], cfg['rnn_type'], cfg['dropout_rate'],
        cfg['is_bidirectional'], cfg['use_attention']
    ).to(device)

    if not os.path.exists(model_path):
        print("❌ No saved model found, starting training.")
        optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])
        criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<pad>'])
        best_acc = 0.0

        for epoch in range(cfg['epochs']):
            model.train()
            total_loss, total_acc = 0, 0
            random.shuffle(train_set)

            for i in range(0, len(train_set), cfg['batch_size']):
                batch = train_set[i:i + cfg['batch_size']]
                src, tgt = create_batch(batch)

                optimizer.zero_grad()
                outputs, attn_weights = model(src, tgt, cfg['teacher_forcing_prob'])

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

            if avg_train_acc > best_acc:
                best_acc = avg_train_acc
                torch.save(model.state_dict(), model_path)
                print(f"💾 Saved new best model at epoch {epoch + 1} with accuracy {best_acc:.2f}%")

        print(f"Best model saved with accuracy: {best_acc:.2f}%")

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("✅ Loaded saved model from disk.")
    model.eval()

    num_plots = 0
    max_plots = 10
    images = []

    with torch.no_grad():
        for i in range(0, len(test_set), cfg['batch_size']):
            batch = test_set[i:i + cfg['batch_size']]
            src, tgt = create_batch(batch)

            outputs, attn_weights = model(src, tgt, teacher_forcing_ratio=0.0)
            preds = outputs.argmax(-1)

            for j in range(src.size(0)):
                input_seq = ''.join([idx2src.get(idx.item(), '') for idx in src[j] if idx.item() not in [src_vocab['<pad>'], src_vocab['<eos>']]])
                target_seq = ''.join([idx2tgt.get(idx.item(), '') for idx in tgt[j][1:] if idx.item() not in [tgt_vocab['<pad>'], tgt_vocab['<eos>']]])
                pred_seq = ''.join([idx2tgt.get(idx.item(), '') for idx in preds[j][1:] if idx.item() not in [tgt_vocab['<pad>'], tgt_vocab['<eos>']]])

                if num_plots < max_plots:
                    input_tokens = [idx2src.get(idx.item(), '') for idx in src[j] if idx.item() not in [src_vocab['<pad>'], src_vocab['<eos>']]]
                    output_tokens = [idx2tgt.get(idx.item(), '') for idx in preds[j][1:] if idx.item() not in [tgt_vocab['<pad>'], tgt_vocab['<eos>']]]
                    attn_matrix = attn_weights[j].cpu().numpy()

                    wandb_img = plot_attention(attn_matrix, input_tokens, output_tokens, input_seq, pred_seq, num_plots + 1)
                    images.append(wandb_img)
                    num_plots += 1

                if num_plots >= max_plots:
                    break
            if num_plots >= max_plots:
                break

    if images:
        wandb.log({"attention_heatmaps": images})
        wandb.finish()

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
attention_heatmaps(parameters,"best_attention_model.pt")