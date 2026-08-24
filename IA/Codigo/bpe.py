import tiktoken

class BPETokenizer:
    """Encapsulamento do tokenizador BPE utilizando a biblioteca tiktoken."""
    def __init__(self, model_name: str = "gpt2"):
        self.tokenizer = tiktoken.get_encoding(model_name)

    def encode(self, text: str, allowed_special: set = {"<|endoftext|>"}) -> list[int]:
        """Converte texto bruto em Token IDs via BPE."""
        return self.tokenizer.encode(text, allowed_special=allowed_special)

    def decode(self, ids: list[int]) -> str:
        """Decodifica Token IDs para o texto correspondente."""
        return self.tokenizer.decode(ids)

    @property
    def vocab_size(self) -> int:
        """Retorna o tamanho total do vocabulário do modelo."""
        return self.tokenizer.n_vocab