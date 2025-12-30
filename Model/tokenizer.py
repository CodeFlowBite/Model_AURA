import re
import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional
from collections import Counter


class ChatTokenizer:
    """
    Tokenizador para chatbots:
    """
    
    def __init__(self, special_tokens: Optional[List[str]] = None):
        """
        Args:
            special_tokens: Lista de tokens especiales. Por defecto incluye tokens del sistema:
                           - Control: <PAD>, <UNK>, <USER>, <EOS>
                           - Intención: <INTENT>
                           - Bot: <BOT>
                           - Variables: <VAR_001> hasta <VAR_999>, <VAR_RES>

        """
        if special_tokens is None:
            # Tokens base del sistema
            self.special_tokens = [
                '<PAD>', '<UNK>', '<USER>', '<EOS>',
                '<INTENT>', '<STATE>', '</STATE>', '<TAG>', '</TAG>', '<HALT_GENERATION/>', '<BOT>', '<ACTION>', '</ACTION>'
            ]
            # Agregar tokens de variables (VAR_001 hasta VAR_999)
            for i in range(1, 1000):
                self.special_tokens.append(f'<VAR_{i:03d}>')
            # Agregar tokens de variables (RES_001 hasta RES_200)
            for i in range(1, 200):
                self.special_tokens.append(f'<RES_{i:03d}>')
        else:
            self.special_tokens = special_tokens
            
        self.stoi: Dict[str, int] = {}  # string to index
        self.itos: Dict[int, str] = {}  # index to string
        self.vocab_size = 0
        
    def tokenize(self, text: str) -> List[str]:
        # Primero, protegemos los tokens especiales reemplazándolos temporalmente
        protected_text = text
        special_token_map = {}
        
        for i, token in enumerate(self.special_tokens):
            placeholder = f"__SPECIAL_{i}__"
            special_token_map[placeholder] = token
            protected_text = protected_text.replace(token, placeholder)
        
        # Patrón que separa:
        # - Paréntesis: ( )
        # - Signos de interrogación y admiración: ¿ ? ¡ !
        # - Puntuación: . , ; : " ' -
        # - Otros símbolos comunes
        # Mantiene palabras con tildes y ñ juntas
        pattern = r'([¿?¡!().,;:\"\'\-…])|(\s+)|([^\s¿?¡!().,;:\"\'\-…]+)'
        
        tokens = []
        for match in re.finditer(pattern, protected_text):
            token = match.group(0)
            # Ignorar espacios en blanco
            if not token.strip():
                continue
            tokens.append(token)
        
        # Restaurar tokens especiales
        final_tokens = []
        for token in tokens:
            if token in special_token_map:
                final_tokens.append(special_token_map[token])
            else:
                final_tokens.append(token)
        
        return final_tokens
    
    def build_vocab(self, texts: List[str], min_freq: int = 1) -> None:
        # Contar frecuencias de tokens
        token_counter = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            token_counter.update(tokens)
        
        # Inicializar vocabulario con tokens especiales
        self.stoi = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.itos = {idx: token for idx, token in enumerate(self.special_tokens)}
        
        # Agregar tokens que cumplan con la frecuencia mínima
        idx = len(self.special_tokens)
        for token, freq in token_counter.most_common():
            if freq >= min_freq and token not in self.stoi:
                self.stoi[token] = idx
                self.itos[idx] = token
                idx += 1
        
        self.vocab_size = len(self.stoi)
        print(f"Vocabulario construido con {self.vocab_size} tokens")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = self.tokenize(text)
        
        # Agregar tokens especiales de inicio y fin
        if add_special_tokens:
            tokens = ['<USER>'] + tokens + ['<EOS>']
        
        # Convertir tokens a índices (usar <UNK> para tokens desconocidos)
        unk_idx = self.stoi.get('<UNK>', 1)
        indices = [self.stoi.get(token, unk_idx) for token in tokens]
        
        return indices
    
    def decode(self, indices: List[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        for idx in indices:
            token = self.itos.get(idx, '<UNK>')
            
            # Omitir tokens especiales si se solicita
            if skip_special_tokens and token in self.special_tokens:
                continue
            
            tokens.append(token)
        
        # Unir tokens con espacios, pero sin espacio antes de puntuación
        text = ''
        for i, token in enumerate(tokens):
            # No agregar espacio antes de puntuación o al inicio
            if i == 0 or token in '.,;:!?)¿¡':
                text += token
            # No agregar espacio después de signos de apertura
            elif i > 0 and tokens[i-1] in '(¿¡':
                text += token
            else:
                text += ' ' + token
        
        return text.strip()
    
    def __len__(self) -> int:
        """Retorna el tamaño del vocabulario"""
        return self.vocab_size
    
    def save(self, filepath: str) -> None:
        data = {
            'special_tokens': self.special_tokens,
            'stoi': self.stoi,
            'itos': {int(k): v for k, v in self.itos.items()},  # Convertir keys a int para JSON
            'vocab_size': self.vocab_size
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Tokenizador guardado en {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'ChatTokenizer':
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Crear tokenizador
        tokenizer = ChatTokenizer(special_tokens=data['special_tokens'])
        
        # Restaurar vocabulario
        tokenizer.stoi = data['stoi']
        tokenizer.itos = {int(k): v for k, v in data['itos'].items()}
        tokenizer.vocab_size = data['vocab_size']
        
        print(f"Tokenizador cargado desde {filepath} (vocab_size: {tokenizer.vocab_size})")
        return tokenizer


class GPTChatDataset(Dataset):
    """
    Dataset de PyTorch para entrenar modelos GPT-style.
    Convierte conversaciones en un formato de secuencia única con tokens especiales.
    """
    def __init__(
        self,
        json_path: str,
        tokenizer: ChatTokenizer,
        max_length: Optional[int] = 512,
        separator: str = " "
    ):
        """
        Args:
            json_path: Ruta al archivo JSON con formato [{"x": "pregunta", "y": "respuesta"}, ...]
            tokenizer: Instancia de ChatTokenizer
            max_length: Longitud máxima de secuencia
            separator: Separador entre input y output (por defecto un espacio)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.separator = separator
        
        # Cargar datos del JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"Cargadas {len(self.data)} conversaciones desde {json_path}")
    
    def format_conversation(self, x: str, y: str) -> str:
        """
        Convierte una conversación en un string plano formato GPT.
        Solo la respuesta es el objetivo de entrenamiento.
        
        Args:
            x: Texto de entrada (pregunta)
            y: Texto de salida (respuesta)
            
        Returns:
            String formateado para entrenamiento GPT
        """
        # Formato simple: entrada + respuesta en una sola secuencia
        formatted = f"{x}{self.separator}{y}"
        return formatted
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retorna un item del dataset en formato GPT.
        
        Returns:
            Diccionario con:
            - input_ids: Secuencia completa tokenizada
            - labels: Solo la respuesta (resto con -100 para ignorar en loss)
        """
        # Obtener conversación
        conversation = self.data[idx]
        x = conversation['x']
        y = conversation['y']
        
        # Tokenizar pregunta y respuesta por separado
        x_tokens = self.tokenizer.tokenize(x)
        y_tokens = self.tokenizer.tokenize(y)
        
        # Convertir a índices
        unk_idx = self.tokenizer.stoi.get('<UNK>', 1)
        x_ids = [self.tokenizer.stoi.get(token, unk_idx) for token in x_tokens]
        y_ids = [self.tokenizer.stoi.get(token, unk_idx) for token in y_tokens]
        
        # Agregar tokens especiales
        sos_id = self.tokenizer.stoi['<USER>']
        eos_id = self.tokenizer.stoi['<EOS>']
        
        # Formato: <USER> pregunta <EOS> respuesta <EOS>
        input_ids = [sos_id] + x_ids + [eos_id] + y_ids + [eos_id]
        
        # Labels: ignorar todo menos la respuesta
        # -100 le dice a PyTorch que ignore estos tokens en el cálculo del loss
        labels = [-100] * (len(x_ids) + 2) + y_ids + [eos_id]  # +2 por <USER> y primer <EOS>
        
        # Truncar si es necesario
        if self.max_length and len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
        
        return {
            'input_ids': torch.tensor(input_ids[:-1], dtype=torch.long),  # Todo menos el último token
            'labels': torch.tensor(labels[1:], dtype=torch.long),  # Todo menos el primero (shifted)
            'seq_length': len(input_ids) - 1
        }


def collate_fn_gpt(batch: List[Dict[str, torch.Tensor]], pad_idx: int = 0) -> Dict[str, torch.Tensor]:
    """
    Función de collate para batches de GPT con padding.
    
    Args:
        batch: Lista de items del dataset
        pad_idx: Índice del token de padding
        
    Returns:
        Diccionario con tensores con padding aplicado
    """
    # Encontrar longitud máxima en el batch
    max_len = max([item['input_ids'].size(0) for item in batch])
    
    batch_size = len(batch)
    
    # Inicializar tensores con padding
    input_ids = torch.full((batch_size, max_len), pad_idx, dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)  # -100 para ignorar
    
    # Llenar tensores
    for i, item in enumerate(batch):
        seq_len = item['input_ids'].size(0)
        input_ids[i, :seq_len] = item['input_ids']
        labels[i, :seq_len] = item['labels']
    
    return {
        'input_ids': input_ids,
        'labels': labels
    }


def create_dataloader(json_path: str, tokenizer: ChatTokenizer, block_size: int, 
                     batch_size: int = 8, shuffle: bool = True):
    """
    Args:
        json_path: Ruta al archivo JSON
        tokenizer: Instancia de ChatTokenizer
        block_size: Tamaño máximo de secuencia
        batch_size: Tamaño del batch
        shuffle: Si mezclar los datos
        
    Returns:
        DataLoader configurado
    """
    from torch.utils.data import DataLoader
    
    dataset = GPTChatDataset(
        json_path=json_path,
        tokenizer=tokenizer,
        max_length=block_size
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_fn_gpt(batch, pad_idx=tokenizer.stoi['<PAD>'])
    )
    
    return dataloader

