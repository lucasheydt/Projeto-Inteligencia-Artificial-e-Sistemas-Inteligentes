import torch
from avaliacao import calc_loss_batch, calc_loss_loader

def train_model(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter):
    """
    Executa o loop de treinamento iterando por épocas e lotes, 
    realizando a retropropagação e atualização dos pesos.
    """
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            # 1. Zera os gradientes do passo anterior
            optimizer.zero_grad()
            
            # 2. Calcula a função de perda (Forward pass)
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            
            # 3. Retropropagação dos gradientes (Backward pass)
            loss.backward()
            
            # 4. Atualização dos pesos
            optimizer.step()
            
            tokens_seen += input_batch.numel()
            global_step += 1
            
            # Avaliação periódica do modelo
            if global_step % eval_freq == 0:
                model.eval() # Desativa dropout para avaliação
                with torch.no_grad():
                    train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
                    val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                
                print(f"Época {epoch+1:02d} | Step {global_step:04d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                model.train() # Retorna ao modo de treinamento

    return train_losses, val_losses, track_tokens_seen