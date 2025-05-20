import torch
from torch.nn.utils.rnn import pad_sequence


def read_pairs(file_path):
    with open(file_path, encoding='utf-8') as f:
        return [(line.split('\t')[1], line.split('\t')[0]) for line in f.read().strip().split('\n') if '\t' in line]

def build_vocab_and_prepare_batch(seqs, device):
    special_tokens = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
    
    # Build character sets
    unique_chars_latin = sorted(set(ch for seq in seqs for ch in seq[0]))
    unique_chars_dev = sorted(set(ch for seq in seqs for ch in seq[1]))

    # Build vocabularies
    src_vocab = {ch: idx + len(special_tokens) for idx, ch in enumerate(unique_chars_latin)}
    tgt_vocab = {ch: idx + len(special_tokens) for idx, ch in enumerate(unique_chars_dev)}
    src_vocab.update(special_tokens)
    tgt_vocab.update(special_tokens)

    idx2src = {idx: ch for ch, idx in src_vocab.items()}
    idx2tgt = {idx: ch for ch, idx in tgt_vocab.items()}

    def encode_text(seq, vocab):
        return [vocab.get(ch, vocab['<unk>']) for ch in seq]

    def create_batch(pairs):
        src = [torch.tensor(encode_text(x, src_vocab) + [src_vocab['<eos>']]) for x, _ in pairs]
        tgt = [torch.tensor([tgt_vocab['<sos>']] + encode_text(y, tgt_vocab) + [tgt_vocab['<eos>']]) for _, y in pairs]
        src = pad_sequence(src, batch_first=True, padding_value=src_vocab['<pad>'])
        tgt = pad_sequence(tgt, batch_first=True, padding_value=tgt_vocab['<pad>'])
        return src.to(device), tgt.to(device)

    return src_vocab, idx2src, tgt_vocab, idx2tgt, create_batch, unique_chars_latin, unique_chars_dev

def compute_word_level_accuracy(preds, targets, vocab):
    sos, eos, pad = vocab['<sos>'], vocab['<eos>'], vocab['<pad>']
    preds = preds.tolist()
    targets = targets.tolist()
    correct = 0
    for p, t in zip(preds, targets):
        p = [x for x in p if x != pad and x != eos]
        t = [x for x in t if x != pad and x != eos]
        if p == t:
            correct += 1
    return correct / len(preds) * 100