import re

class SimpleTokenizerV1:
    """Tokenizador simples baseado em Expressões Regulares e Vocabulário dinâmico."""
    def __init__(self, vocab: dict):
        self.str_to_id = vocab
        self.id_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        """Converte texto bruto em uma lista de Token IDs."""
        tokens = re.split(r'([,.?_!"()\']|--|\s+)', text)
        tokens = [item.strip() for item in tokens if item.strip()]
        return [self.str_to_id[token] for token in tokens]

    def decode(self, ids: list[int]) -> str:
        """Converte uma lista de Token IDs de volta para texto."""
        text = " ".join([self.id_to_str[i] for i in ids])
        # Ajusta pontuações grudadas
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


class SimpleTokenizerV2:
    """Tokenizador com suporte a tokens especiais de contexto e palavras desconhecidas (<|unk|> e <|endoftext|>)."""
    def __init__(self, vocab: dict):
        self.str_to_id = vocab
        self.id_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        """Converte texto bruto tratando tokens ausentes no vocabulário."""
        tokens = re.split(r'([,.?_!"()\']|--|\s+)', text)
        tokens = [item.strip() for item in tokens if item.strip()]
        tokens = [
            item if item in self.str_to_id else "<|unk|>"
            for item in tokens
        ]
        return [self.str_to_id[token] for token in tokens]

    def decode(self, ids: list[int]) -> str:
        """Converte Token IDs em texto recuperado."""
        text = " ".join([self.id_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text