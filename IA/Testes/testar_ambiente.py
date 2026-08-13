import sys
import torch
import tiktoken

def executar_testes():
    print("=" * 50)
    print("      VALIDAÇÃO DO AMBIENTE DE DESENVOLVIMENTO      ")
    print("=" * 50)

    # 1. Verificar se o .venv está ativo
    caminho_python = sys.executable
    print(f"\n[1/4] Executável Python:")
    print(f"      {caminho_python}")
    if ".venv" in caminho_python:
        print("      ✅ STATUS: Rodando DENTRO do ambiente virtual (.venv)!")
    else:
        print("      ⚠️ STATUS: Atenção! Não está rodando no .venv.")

    # 2. Testar PyTorch
    print(f"\n[2/4] Testando PyTorch (Versão: {torch.__version__}):")
    try:
        # Criando e somando tensores simples
        tensor_a = torch.tensor([1.0, 2.0, 3.0])
        tensor_b = torch.tensor([4.0, 5.0, 6.0])
        resultado = tensor_a + tensor_b
        print(f"      Soma de tensores: {tensor_a.tolist()} + {tensor_b.tolist()} = {resultado.tolist()}")
        print("      ✅ STATUS: PyTorch funcionando perfeitamente!")
    except Exception as e:
        print(f"      ❌ ERRO no PyTorch: {e}")

    # 3. Testar Tiktoken
    print(f"\n[3/4] Testando Tiktoken (Versão: {tiktoken.__version__}):")
    try:
        # Tokenizando um texto usando o vocab do GPT-2
        enc = tiktoken.get_encoding("gpt2")
        texto = "Construindo um LLM do zero!"
        tokens = enc.encode(texto)
        print(f"      Texto: '{texto}'")
        print(f"      Tokens gerados: {tokens}")
        print("      ✅ STATUS: Tiktoken funcionando perfeitamente!")
    except Exception as e:
        print(f"      ❌ ERRO no Tiktoken: {e}")

    # 4. Testar Jupyter
    print(f"\n[4/4] Checando Suporte ao Jupyter:")
    try:
        import jupyter
        print("      ✅ STATUS: Jupyter instalado e pronto para notebooks!")
    except ImportError:
        print("      ⚠️ Jupyter ainda não instalado. (Rode 'pip install jupyter' se for usar .ipynb)")

    print("\n" + "=" * 50)
    print("         TODOS OS TESTES FORAM CONCLUÍDOS!          ")
    print("=" * 50)

if __name__ == "__main__":
    executar_testes()