import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import TransliterationModel
import torch
import torch.nn as nn
import torch.optim as optim
import random
from utils.helper_functions import build_vocab_and_prepare_batch, compute_word_level_accuracy, read_pairs
import wandb

def generate_sentence_html(words_data):
    """
    words_data is a list of dictionaries, one per word.
    Each dictionary should have:
      - 'input_chars': list of input characters (Latin)
      - 'output_chars': list of output characters (native script)
      - 'attention_weights': list of lists (each inner list the attention weights for corresponding output character)
    """
    html_template = f"""
    <html>
    <head>
      <meta charset="UTF-8">
      <style>
        body {{
          font-family: Arial, sans-serif;
          padding: 20px;
        }}
        .top-attn-boxes {{
          display: flex;
          gap: 20px;
          margin-bottom: 20px;
        }}
        .attn-box {{
          border: 1px solid #ccc;
          padding: 10px;
          width: 150px;
          height: 50px;
          text-align: center;
          font-size: 18px;
          background-color: #f9f9f9;
          border-radius: 8px;
          display: flex;
          align-items: center;
          justify-content: center;
        }}
        .sentence {{
          line-height: 2;
          font-size: 28px;
        }}
        .word {{
          margin-right: 20px; /* proper spacing between words */
          display: inline-block;
        }}
        .output-char {{
          display: inline-block;
          margin: 0 3px;
          padding: 8px 5px;
          cursor: pointer;
          border-bottom: 1px dotted #555;
          transition: background-color 0.2s;
        }}
        .output-char:hover {{
          background-color: #eef;
        }}
      </style>
    </head>
    <body>
      <div class="top-attn-boxes">
        <div class="attn-box" id="top1">—</div>
        <div class="attn-box" id="top2">—</div>
        <div class="attn-box" id="top3">—</div>
      </div>
      <div class="sentence">
    """
    # For each word in the sentence
    for word in words_data:
        input_chars = word["input_chars"]
        output_chars = word["output_chars"]
        attention_weights = word["attention_weights"]
        word_html = '<span class="word">'
        for i, out_char in enumerate(output_chars):
            # Prepare data: list of dicts with char and weight for this output char.
            data = [
                {"char": input_chars[j], "weight": round(w, 3)}
                for j, w in enumerate(attention_weights[i])
            ]
            # Encode the data into a HTML-friendly format.
            data_str = str(data).replace("'", "&quot;")
            # If an output character is empty, we show a placeholder (like ␣)
            display_char = out_char if out_char else "␣"
            word_html += f'<span class="output-char" data-attn="{data_str}">{display_char}</span>'
        word_html += '</span>'  # Close the word span.
        html_template += word_html

    html_template += """
      </div>
      <script>
        function showTop3(attnData) {
          // Create a shallow copy to sort so we don't modify the original array
          let sortedData = attnData.slice().sort((a, b) => b.weight - a.weight);
          const top = sortedData.slice(0, 3);
          for (let i = 0; i < 3; i++) {
            const el = document.getElementById("top" + (i + 1));
            if (i < top.length) {
              el.innerText = top[i].char + " : " + top[i].weight.toFixed(2);
            } else {
              el.innerText = "—";
            }
          }
        }

        // Attach hover event to each character.
        document.querySelectorAll(".output-char").forEach(span => {
          span.addEventListener("mouseenter", () => {
            // Replace HTML entity quotes with actual quotes and parse the JSON data.
            const attn = JSON.parse(span.dataset.attn.replace(/&quot;/g, '"'));
            showTop3(attn);
          });
        });
      </script>
    </body>
    </html>
    """
    return html_template

def Interactive_plot(cfg, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    wandb.finish()
    wandb.init(
        project="transliteration_attention_Interactive_plot",
        name='connectivity_plot',
        resume="never",
        reinit=True,
        config=cfg
    )

    # Load training and test datasets
    train_path = "dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv"
    train_set = read_pairs(train_path)
    test_path = "dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv"
    test_set = read_pairs(test_path)

    # Prepare vocabulary and batch creation
    src_vocab, idx2src, tgt_vocab, idx2tgt, create_batch, tensor_to_words, _ = build_vocab_and_prepare_batch(train_set, device)

    # Initialize the model
    model = TransliterationModel(
        len(src_vocab), len(tgt_vocab), cfg['embedding_size'], cfg['hidden_size'],
        cfg['enc_layers'], cfg['dec_layers'], cfg['rnn_type'], cfg['dropout_rate'],
        cfg['is_bidirectional'], cfg['use_attention']
    ).to(device)

    # If model not trained yet, train it
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

    # Load and evaluate saved model
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ Loaded saved model from disk.")

    model.eval()
    num_plots = 0
    max_plots = 10
    word_data=[]
    random.shuffle(test_set)
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
                        # Define special tokens to exclude
                    special_tokens = {'<pad>', '<sos>', '<eos>'}
                    filtered_input_tokens = [tok for tok in input_tokens if tok not in special_tokens]
                    filtered_output_tokens = [tok for tok in output_tokens if tok not in special_tokens]
                    attn_matrix_filtered = attn_matrix[:len(filtered_output_tokens), :len(filtered_input_tokens)]
                    num_plots += 1
                    pred_dict = {
                        "input_chars": filtered_input_tokens,
                        "output_chars": filtered_output_tokens,
                        "attention_weights": attn_matrix_filtered.tolist(),
                    }
                    word_data.append(pred_dict)
                else:
                    break
            if num_plots >= max_plots:
                break
    # Generate HTML
    html_str = generate_sentence_html(word_data)

    # Save the HTML file with UTF-8 encoding
    with open("sentence_attention.html", "w", encoding="utf-8") as f:
        f.write(html_str)

    wandb.log({"sentence_attention_viz": wandb.Html("sentence_attention.html", inject=False)})
    wandb.finish()

    print("HTML file generated and logged to wandb (if wandb is configured).")

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
Interactive_plot(parameters,"best_attention_model.pt")