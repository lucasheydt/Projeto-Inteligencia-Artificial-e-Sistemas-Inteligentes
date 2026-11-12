import torch
import torch.nn.functional as F
import math

def calc_loss_batch(input_batch, target_batch, model, device):
    """Calcula a perda (Cross Entropy) para um único lote de dados."""
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    
    # Forward pass: o modelo gera os logits (previsões não normalizadas)
    logits = model(input_batch)
    
    # Achata os tensores porque o F.cross_entropy do PyTorch espera [batch_size * seq_len, vocab_size]
    logits_flat = logits.flatten(0, 1)
    targets_flat = target_batch.flatten()
    
    loss = F.cross_entropy(logits_flat, targets_flat)
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    """Calcula a perda média sobre os conjuntos de treinamento ou validação."""
    total_loss = 0.
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
        
    if num_batches == 0:
        return float("nan")

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
            
    return total_loss / num_batches

def evaluate_metrics(model, train_loader, val_loader, device):
    """Mede a perda e calcula a perplexidade nos conjuntos de treino e validação."""
    model.eval() # Desativa camadas de regularização como Dropout
    
    with torch.no_grad(): # Desativa o cálculo de gradientes para economizar memória
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)
    
    model.train() # Retorna ao modo de treinamento
    
    # Perplexity é a exponencial da Cross-Entropy Loss
    try:
        train_perplexity = math.exp(train_loss)
    except OverflowError:
        train_perplexity = float('inf')
        
    try:
        val_perplexity = math.exp(val_loss)
    except OverflowError:
        val_perplexity = float('inf')
    
    return train_loss, val_loss, train_perplexity, val_perplexity

