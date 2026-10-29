import torch

def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    Geração autorregressiva simples utilizando amostragem gulosa (Greedy Decoding / Argmax).
    """
    model.eval() # Desativa o dropout durante a geração
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Foca apenas no último token da sequência
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
        
    return idx


def generate_text_advanced(model, idx, max_new_tokens, context_size, temperature=1.0, top_k=None):
    """
    Geração autorregressiva com controle de Temperatura e Top-K Sampling.
    
    - Temperature: Escala os logits. Val < 1.0 torna a geração mais determinística; Val > 1.0 mais criativa.
    - Top-K: Filtra apenas os K tokens com maior probabilidade antes da amostragem.
    """
    model.eval() # Desativa o dropout durante a geração
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :]
        
        # 1. Aplicação da Temperatura
        if temperature > 0.0:
            logits = logits / temperature
        
        # 2. Filtragem Top-K
        if top_k is not None and top_k > 0:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        
        # 3. Conversão para Probabilidades e Amostragem Multinomial
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probas, num_samples=1)
        
        idx = torch.cat((idx, idx_next), dim=1)
        
    return idx