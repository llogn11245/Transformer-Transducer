import pandas as pd

# Đọc file train và dev
train = pd.read_csv('train.csv')
dev = pd.read_csv('dev.csv')

# Gộp transcript của cả train và dev
all_transcripts = pd.concat([train['transcript'], dev['transcript']])

# Tách từ và loại bỏ trùng lặp
vocab = set()
for sentence in all_transcripts:
    words = sentence.strip().split()
    vocab.update(words)

# Sắp xếp từ điển
vocab = sorted(vocab)

# Ghi ra file vocab.txt
with open('vocab.txt', 'w', encoding='utf-8') as f:
    f.write("<pad>\n")
    f.write("<sos>\n")
    f.write("<eos>\n")
    f.write("<unk>\n")
    f.write("<blank>\n")
    for word in vocab:
        f.write(f'{word}\n')
