from torch.utils.data import Dataset, DataLoader
import torch
import json
import re
from collections import Counter
from typing import Optional, List, Dict


class SimpleTokenizer:
    """Tokenizador por palabras diseñado para español.

    - Convierte a minúsculas
    - Separa signos de puntuación definidos en una lista manual
    - Conserva acentos y la letra ñ
    - Elimina espacios extra
    """

    def __init__(self, extra_punct: Optional[List[str]] = None):
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}
        self.special_tokens = ['<PAD>', '<UNK>', '<EOS>', '<USER>', '<BOT>', '<INTENT>', '<CODE>', '</CODE>', '<RESULT>', '</RESULT>', '<STEPS>', '</STEPS>', '<EXECUTE>', '<EXPLAIN>']

        # puntuación manual (no escapada): se separará de las palabras
        base_punct = ['.', ',', '!', '?', '¿', '¡', ':', ';', '(', ')', '[', ']', '{', '}', '"', "'", '…', '|']
        self.punctuation = base_punct + (extra_punct or [])

        # construir regex para separar puntuación; usamos lookahead/lookbehind para no perder caracteres
        escaped = [re.escape(p) for p in self.punctuation]
        # patrón que encuentra cada signo de puntuación
        self.punct_regex = re.compile('(' + '|'.join(escaped) + ')')
        self.digit_regex = re.compile(r'(\d)')

    def tokenize(self, text: str) -> List[str]:
        text = text.lower().strip()

        # separar puntuación
        text = self.punct_regex.sub(r' \1 ', text)

        # separar cada dígito
        text = self.digit_regex.sub(r' \1 ', text)

        # normalizar espacios
        tokens = re.split(r'\s+', text)
        tokens = [t for t in tokens if t]

        return tokens

    def build_vocab(self, texts: List[str], min_freq: int = 1):
        counter = Counter()
        if texts:
            for text in texts:
                counter.update(self.tokenize(text))

        vocab = [w for w, c in counter.items() if c >= min_freq]
        vocab = sorted(vocab)
        all_tokens = self.special_tokens + vocab
        self.stoi = {w: i for i, w in enumerate(all_tokens)}
        self.itos = {i: w for w, i in self.stoi.items()}

    def encode(self, text: str, add_eos: bool = True) -> List[int]:
        tokens = self.tokenize(text)
        if add_eos:
            tokens.append('<EOS>')
        return [self.stoi.get(t, self.stoi['<UNK>']) for t in tokens]

    def decode(self, indices: List[int]) -> str:
        words = [self.itos.get(i, '<UNK>') for i in indices]
        words = [w for w in words if w not in self.special_tokens]
        out = []
        for w in words:
            if w in self.punctuation:
                if out:
                    out[-1] = out[-1] + w
                else:
                    out.append(w)
            else:
                out.append(w)
        return ' '.join(out)

class ChatDataset(Dataset):
    def __init__(self, json_path: str, tokenizer: SimpleTokenizer, block_size: int):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.samples = []
        for item in data:
            q, a = item['x'], item['y']
            pair = f"{q} => {a}"
            ids = tokenizer.encode(pair)
            if len(ids) > block_size:
                ids = ids[:block_size]
            self.samples.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx][:-1]
        y = self.samples[idx][1:]
        return x, y


def create_dataloader(json_path: str, tokenizer: SimpleTokenizer, block_size: int, batch_size: int = 4, collate_fn = None):
    dataset = ChatDataset(json_path, tokenizer, block_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
