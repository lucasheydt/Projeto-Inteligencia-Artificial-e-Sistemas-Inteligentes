import torch
import torch.nn as nn

class GPTInputEmbeddings(nn.Module):
    """Combina Token Embeddings com Positional Embeddings para alimentar o modelo."""
    def __init__(self, vocab_size: int, output_dim: int, max_length: int):
        super().__init__()
        self.token_embedding_layer = nn.Embedding(vocab_size, output_dim)
        self.pos_embedding_layer = nn.Embedding(max_length, output_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Recebe um lote de IDs de forma (batch_size, seq_len)
        Retorna vetores combinados de forma (batch_size, seq_len, output_dim)
        """
        batch_size, seq_len = input_ids.shape
        
        # Gera embeddings dos tokens
        token_embeddings = self.token_embedding_layer(input_ids)
        
        # Gera embeddings de posição baseados no tamanho da sequência recebida
        pos_ids = torch.arange(seq_len, device=input_ids.device)
        pos_embeddings = self.pos_embedding_layer(pos_ids)
        
        # Soma elemento a elemento (Broadcast do vetor de posição no lote)
        input_embeddings = token_embeddings + pos_embeddings
        return input_embeddings