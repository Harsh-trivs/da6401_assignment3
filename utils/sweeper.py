from model import TransliterationModel
import torch
import random
from utils.helper_functions import build_vocab_and_prepare_batch, compute_word_level_accuracy, read_pairs
import wandb
import torch.nn as nn
import torch.optim as optim

def run_training():
    # Initialize wandb config
    wandb.init()
    cfg = wandb.config
    wandb.run.name = (
    f"es_{cfg.embedding_size}_hs_{cfg.hidden_size}_"
    f"enc_{cfg.enc_layers}_dec_{cfg.dec_layers}_"
    f"rnn_{cfg.rnn_type}_dropout_{cfg.dropout_rate}_"
    f"bidirectional_{cfg.is_bidirectional}_"
    f"lr_{cfg.learning_rate}_bs_{cfg.batch_size}_"
    f"epochs_{cfg.epochs}_tfp_{cfg.teacher_forcing_prob}_"
    f"beam_size_{cfg.beam_size}"
    )


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and prepare data
    train_path = "dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv"
    dev_path = "dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv"
    train_set = read_pairs(train_path)
    dev_set = read_pairs(dev_path)

    src_vocab, idx2src, tgt_vocab, idx2tgt, create_batch, _, _ = build_vocab_and_prepare_batch(train_set, device)

    # Initialize model, optimizer, criterion
    model = TransliterationModel(
        len(src_vocab), len(tgt_vocab), cfg.embedding_size, cfg.hidden_size,
        cfg.enc_layers, cfg.dec_layers, cfg.rnn_type, cfg.dropout_rate,
        cfg.is_bidirectional,cfg.use_attention
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<pad>'])

    # Training loop
    for epoch in range(cfg.epochs):
        model.train()
        total_loss, total_acc = 0, 0
        random.shuffle(train_set)

        for i in range(0, len(train_set), cfg.batch_size):
            batch = train_set[i:i+cfg.batch_size]
            src, tgt = create_batch(batch)

            optimizer.zero_grad()
            if cfg.use_attention:
                outputs, attn_weights = model(src, tgt, cfg.teacher_forcing_prob)
            else:
                outputs = model(src, tgt, cfg.teacher_forcing_prob)

            loss = criterion(outputs[:, 1:].reshape(-1, outputs.size(-1)), tgt[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(-1)
            acc = compute_word_level_accuracy(preds[:, 1:], tgt[:, 1:], tgt_vocab)

            total_loss += loss.item()
            total_acc += acc

        avg_train_loss = total_loss / (len(train_set) // cfg.batch_size)
        avg_train_acc = total_acc / (len(train_set) // cfg.batch_size)

        # Validation
        model.eval()
        dev_loss, dev_acc = 0, 0
        with torch.no_grad():
            for i in range(0, len(dev_set), cfg.batch_size):
                batch = dev_set[i:i+cfg.batch_size]
                src, tgt = create_batch(batch)
                if cfg.use_attention:
                    outputs, attn_weights = model(src, tgt, 0)
                else:
                    outputs = model(src, tgt, 0,)
                loss = criterion(outputs[:, 1:].reshape(-1, outputs.size(-1)), tgt[:, 1:].reshape(-1))

                preds = outputs.argmax(-1)
                acc = compute_word_level_accuracy(preds[:, 1:], tgt[:, 1:], tgt_vocab)

                dev_loss += loss.item()
                dev_acc += acc

        avg_dev_loss = dev_loss / (len(dev_set) // cfg.batch_size)
        avg_dev_acc = dev_acc / (len(dev_set) // cfg.batch_size)

        # Logging
        wandb.log({
            "Epoch": epoch + 1,
            "Train Loss": avg_train_loss,
            "Train Accuracy": avg_train_acc,
            "Validation Loss": avg_dev_loss,
            "Validation Accuracy": avg_dev_acc,
        })

        print(f"Epoch {epoch+1}/{cfg.epochs} | Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.2f}% | Val Loss: {avg_dev_loss:.4f}, Val Acc: {avg_dev_acc:.2f}%")

    wandb.finish()
    return model
