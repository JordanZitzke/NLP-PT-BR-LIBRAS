from transformers import AutoModelForSeq2SeqLM, T5Tokenizer, pipeline

# Caminho para o modelo treinado
MODEL_DIR = "./my_awesome_pt_libras_model"

# Carregar tokenizador e modelo
tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

# Criar pipeline de tradução
tradutor = pipeline(
    "translation", 
    model=model, 
    tokenizer=tokenizer
)

def traduzir(texto):
    """Traduz um texto de Português para LIBRAS"""
    input_text = f"translate Portuguese to LIBRAS: {texto}"
    resultado = tradutor(input_text, max_length=128)
    return resultado[0]["translation_text"].upper()

# Exemplo de uso
if __name__ == "__main__":
    while True:
        texto = input("\nDigite um texto em português (ou 'sair' para encerrar): ")
        if texto.lower() == 'sair':
            break
        
        traducao = traduzir(texto)
        print(f"LIBRAS: {traducao}")