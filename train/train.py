from Model.tokenizer import ChatTokenizer, create_dataloader
from Model.GPTMini import GPTMini, GPTMiniConfig
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json


def evaluar(model, val_loader, best_loss, device):
    """
    Evalúa el modelo en el conjunto de validación.
    """
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids)
            
            # Calcular pérdida (CrossEntropyLoss ignora automáticamente -100)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            val_loss += loss.item()

    val_loss /= len(val_loader)
    
    if val_loss < best_loss:
        model.save("GPTMINI_chatbot")
        print(f"Nuevo mejor modelo guardado (Val Loss: {val_loss:.4f})")
        return val_loss
    
    model.train()
    return best_loss


def train():
    device =  'cpu'
    print(f"Usando dispositivo: {device}\n")

    tokenizer = ChatTokenizer()
    
    # Cargar datos para construir vocabulario
    with open('data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    texts = []
    for d in data:
        texts.append(d['x'])
        texts.append(d['y'])
    
    # Construir vocabulario
    print("Construyendo vocabulario...")
    tokenizer.build_vocab(texts, min_freq=1)
    vocab_size = len(tokenizer.stoi)
    print(f"Tamaño del vocabulario: {vocab_size}")
    
    # Guardar tokenizador para uso posterior
    tokenizer.save('tokenizer_chatbot.json')
    print("Tokenizador guardado en 'tokenizer_chatbot.json'\n")

    # Configuración del modelo (MÁS PEQUEÑO para evitar overfitting)
    config = GPTMiniConfig(
        vocab_size=vocab_size,
        n_layer=2,
        n_head=2, 
        n_embd=128,
        block_size=256,
        dropout=0.3 
    )

    # Crear modelo
    model = GPTMini(config).to(device)
    print(f"Modelo creado con {sum(p.numel() for p in model.parameters())} parámetros\n")
    
    optimizer = AdamW(
        model.parameters(), 
        lr=3e-4,  
        weight_decay=0.05  
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3
    )

    # Crear dataloaders usando la función auxiliar
    print("Cargando datasets...")
    dataloader = create_dataloader(
        'data.json',
        tokenizer,
        config.block_size,
        batch_size=8,
        shuffle=True
    )
    
    dataloader_eval = create_dataloader(
        'data_eval.json',
        tokenizer,
        config.block_size,
        batch_size=5,
        shuffle=False
    )

    print(f"Batches en train: {len(dataloader)}")
    print(f"Batches en eval: {len(dataloader_eval)}\n")

    # Parámetros de entrenamiento
    model.train()
    epochs = 100
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    # NUEVO: Detección de overfitting
    overfitting_threshold = 0.3  # Si train_loss es 0.3 menos que val_loss, hay overfitting

    print("=== Iniciando entrenamiento ===\n")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            logits = model(input_ids)
            
            # Calcular pérdida (CrossEntropyLoss ignora -100 automáticamente)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping para estabilidad
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()

            total_loss += loss.item()
        
        # Calcular pérdida promedio de entrenamiento
        avg_train_loss = total_loss / len(dataloader)
        
        # Evaluar en validación
        prev_best = best_loss
        best_loss = evaluar(model, dataloader_eval, best_loss, device)
        
        # Actualizar learning rate basado en validación
        scheduler.step(best_loss)
        
        # Early stopping check
        if best_loss < prev_best:
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Imprimir progreso
        current_lr = optimizer.param_groups[0]['lr']
        overfitting_gap = best_loss - avg_train_loss
        
        print(f"Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {best_loss:.4f} | Gap: {overfitting_gap:.4f} | "
              f"LR: {current_lr:.6f} | Patience: {patience_counter}/{patience}")
        
        # Advertencia de overfitting
        if overfitting_gap > overfitting_threshold:
            print(f"⚠️  ADVERTENCIA: Overfitting detectado (Gap: {overfitting_gap:.4f})")
        
        # Parar si no mejora
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping en epoch {epoch+1}")
            print(f"📊 Overfitting gap final: {overfitting_gap:.4f}")
            break

    # Guardar modelo final
    model.save("gptmini_chatbot_final")
    print("\n=== Entrenamiento completado ===")
    print(f"Mejor validation loss: {best_loss:.4f}")
