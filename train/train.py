from Model.tokenizer import *
from Model.GPTMini import GPTMini, GPTMiniConfig
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import time
import json

def collate_fn(batch, pad_id = 0):
    xs, ys = zip(*batch)

    max_len = max(x.size(0) for x in xs)

    padded_x = []
    padded_y = []

    for x, y in zip(xs, ys):
        pad_len = max_len - x.size(0)

        padded_x.append(
            torch.cat([x, torch.full((pad_len,), pad_id)])
        )

        padded_y.append(
            torch.cat([y, torch.full((pad_len,), pad_id)])
        )

    return torch.stack(padded_x), torch.stack(padded_y)

def train():
    # Configuración
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Variables para predecir el tiempo estimado de entrenamiento
    time_init = 0
    time_end = 0
    istime = True

    # Tokenizador
    print("Construyendo vocabulario...")
    texts = []
    with open('data/data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        for d in data:
            texts.append(d['x'])
            texts.append(d['y'])
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(texts)

    #Crear modelo
    max_tokens = len(tokenizer.itos) + 1
    print(f"Cantidad de tokens: {max_tokens}")
    config = GPTMiniConfig(vocab_size=max_tokens, n_layer=8, n_head=6, n_embd=384, block_size=512)
    #config = GPTMiniConfig(vocab_size=max_tokens, n_layer=2, n_head=32, n_embd=128, block_size=512) Etapa inicial de pruebas
    model = GPTMini(config).to(device)

    # Dataset y DataLoader
    dataloader = create_dataloader('data.json', tokenizer, config.block_size, batch_size=100, collate_fn=collate_fn)
    
    # Optimizador
    optimizer = AdamW(model.parameters(), lr=3e-4)

    # Entrenamiento
    print("Entrenando modelo GPTMini...")
    epochs = 1000
    for epoch in range(epochs):
        total_loss = 0
        if istime: time_init = time.time()
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.5f}")
        if istime: 
            time_end = time.time()
            seg = (time_end - time_init) * (epochs - epoch)
            minut = seg / 60
            hours = minut / 60
            print(f'Tiempo estimado en segundos: {seg}, tiempo en minutos: {minut}, tiempo en horas: {hours}')

    # Guardar modelo entrenado
    model.save('gptmini_entrenado')