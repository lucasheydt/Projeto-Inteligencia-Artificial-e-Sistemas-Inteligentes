import torch


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """Gera novos tokens iterativamente (inferência autoregressiva / Greedy Decoding)."""
    # Loop para gerar um token de cada vez
    for _ in range(max_new_tokens):
        # 1. Corta o contexto para caber no tamanho máximo suportado pelo modelo
        idx_cond = idx[:, -context_size:]
        
        # 2. Passa o contexto pelo modelo sem calcular gradientes
        with torch.no_grad():
            logits = model(idx_cond)
        
        # 3. Foca apenas na previsão para o último token da sequência
        logits = logits[:, -1, :]
        
        # 4. Converte os logits em probabilidades e seleciona o token mais provável (argmax)
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        
        # 5. Adiciona o novo token à sequência original para a próxima iteração
        idx = torch.cat((idx, idx_next), dim=1)
        
    return idx

def generate_text_advanced(model, idx, max_new_tokens, context_size, temperature=1.0, top_k=None):
    """
    Gera tokens iterativamente permitindo controle de Temperatura e Top-K Sampling.
    - temperature: Valores < 1.0 deixam o modelo mais conservador. Valores > 1.0 deixam mais criativo/caótico. 0.0 é Greedy.
    - top_k: Limita a amostragem apenas aos K tokens mais prováveis.
    """
    for _ in range(max_new_tokens):
        # Corta o contexto para caber no limite do modelo
        idx_cond = idx[:, -context_size:]
        
        with torch.no_grad():
            logits = model(idx_cond)
            
        # Pega apenas os logits do último token previsto
        logits = logits[:, -1, :]
        
        # 1. Aplica o Top-K (se definido)
        if top_k is not None:
            # Encontra os valores top_k, tudo que for menor que o k-ésimo valor vira -Infinito
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        # 2. Aplica a Temperatura e Amostragem (Sampling)
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            # Escolhe o próximo token probabilisticamente (jogando um dado viciado)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            # Se a temperatura for 0.0, comporta-se exatamente como o Greedy Decoding
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.argmax(probs, dim=-1, keepdim=True)
            
        # Anexa o token gerado à sequência
        idx = torch.cat((idx, idx_next), dim=1)
        
    return idx