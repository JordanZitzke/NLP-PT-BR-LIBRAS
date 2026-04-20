import json
import os
import numpy as np
import time
import torch
from transformers import (
    AutoModelForSeq2SeqLM, 
    T5Tokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq
)
from datasets import Dataset as HFDataset
from tqdm.auto import tqdm
import evaluate

# Configurações
MODEL_CHECKPOINT = "t5-small"  # Modelo base para iniciar o treinamento
DATASET_PATH = "dataset_minusculo.json"
OUTPUT_DIR = "./my_new_pt_libras_model"  # Novo diretório para o novo modelo
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
BATCH_SIZE = 4
MAX_LENGTH = 128
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 5
SAVE_TOTAL_LIMIT = 3
SEED = 42

# Configuração de seed para reprodutibilidade
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True

# Função para carregar o dataset
def load_dataset(dataset_path):
    """Carrega o dataset de um arquivo JSON."""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Converter para o formato esperado pela biblioteca datasets
    processed_data = {
        "id": [],
        "pt": [],
        "lb": []
    }
    
    for item in data:
        processed_data["id"].append(item["id"])
        processed_data["pt"].append(item["translation"]["pt"])
        processed_data["lb"].append(item["translation"]["lb"])
    
    # Criar o dataset usando a biblioteca datasets
    dataset = HFDataset.from_dict(processed_data)
    
    # Dividir em treino e validação (90% treino, 10% validação)
    dataset = dataset.train_test_split(test_size=0.1, seed=SEED)
    
    return dataset

# Funções de pré-processamento
def preprocess_function(examples):
    """Função para pré-processar os exemplos para o modelo T5."""
    # T5 usa o prefixo 'translate' seguido pelos códigos de idioma
    inputs = ["translate Portuguese to LIBRAS: " + text for text in examples["pt"]]
    targets = examples["lb"]
    
    model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, padding="max_length", truncation=True)
    
    # Configurar os targets (labels)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=MAX_LENGTH, padding="max_length", truncation=True)
    
    # Substituir os tokens de padding por -100 para que não contribuam para a loss
    labels_input_ids = labels["input_ids"]
    for idx, label in enumerate(labels_input_ids):
        # Substitui tokens de padding por -100
        labels_input_ids[idx] = [
            -100 if token == tokenizer.pad_token_id else token 
            for token in label
        ]
    
    model_inputs["labels"] = labels_input_ids
    
    return model_inputs

# Função para calcular métricas
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Some simple post-processing - converter para maiúsculas para LIBRAS
    decoded_preds = [pred.strip().upper() for pred in decoded_preds]
    decoded_labels = [label.strip().upper() for label in decoded_labels]
    
    # Registrar alguns exemplos para debugging
    print("\nExemplos para debugging de BLEU:")
    for i in range(min(3, len(decoded_preds))):
        print(f"Pred: {decoded_preds[i]}")
        print(f"Gold: {decoded_labels[i]}")
        print("---")
    
    # SacreBLEU expects a list of references for each prediction
    formatted_labels = [[label] for label in decoded_labels]
    
    # Calculate BLEU score
    result = metric.compute(predictions=decoded_preds, references=formatted_labels)
    
    # Exact match score (também em maiúsculas)
    exact_match = sum([1 if pred == ref[0] else 0 for pred, ref in zip(decoded_preds, formatted_labels)]) / len(decoded_preds)
    
    result = {"bleu": result["score"], "exact_match": exact_match}
    
    return result

# Função para verificar a existência do último checkpoint
def get_last_checkpoint(checkpoint_dir):
    """Verifica se existe um checkpoint e retorna o caminho para o último checkpoint."""
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Verifica os diretórios dentro do diretório de checkpoints
    checkpoint_dirs = [
        os.path.join(checkpoint_dir, d) 
        for d in os.listdir(checkpoint_dir) 
        if os.path.isdir(os.path.join(checkpoint_dir, d)) and d.startswith('checkpoint-')
    ]
    
    if not checkpoint_dirs:
        return None
    
    # Ordena os diretórios por número do checkpoint
    checkpoint_dirs = sorted(
        checkpoint_dirs,
        key=lambda x: int(os.path.basename(x).split('-')[-1])
    )
    
    # Retorna o caminho para o último checkpoint
    last_checkpoint = checkpoint_dirs[-1]
    print(f"Encontrado último checkpoint: {last_checkpoint}")
    return last_checkpoint

def main():
    # Verificar disponibilidade da GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Carregar o dataset
    print(f"Carregando dataset de {DATASET_PATH}...")
    start_time = time.time()
    dataset = load_dataset(DATASET_PATH)
    print(f"Dataset carregado em {time.time() - start_time:.2f} segundos.")
    print(f"Tamanho do conjunto de treino: {len(dataset['train'])}")
    print(f"Tamanho do conjunto de validação: {len(dataset['test'])}")
    
    # Mostrar um exemplo
    print("\nExemplo do dataset:")
    example = dataset['train'][0]
    print(f"ID: {example['id']}")
    print(f"Português: {example['pt']}")
    print(f"LIBRAS: {example['lb']}")
    
    # Processar o dataset
    print("\nProcessando o dataset...")
    start_time = time.time()
    
    # Filtrar exemplos muito longos que podem causar problemas
    def filter_long_examples(example):
        # Verificar se o comprimento dos textos está dentro de limites razoáveis
        return (len(example["pt"]) <= MAX_LENGTH * 4 and len(example["lb"]) <= MAX_LENGTH * 4)
    
    print("Filtrando exemplos muito longos...")
    original_train_size = len(dataset["train"])
    original_test_size = len(dataset["test"])
    
    filtered_dataset = dataset.filter(
        filter_long_examples,
        desc="Filtrando exemplos longos"
    )
    
    print(f"Tamanho do treino após filtragem: {len(filtered_dataset['train'])} (removidos {original_train_size - len(filtered_dataset['train'])} exemplos)")
    print(f"Tamanho do teste após filtragem: {len(filtered_dataset['test'])} (removidos {original_test_size - len(filtered_dataset['test'])} exemplos)")
    
    # Aplicar o pré-processamento
    tokenized_datasets = filtered_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=filtered_dataset["train"].column_names,
        desc="Tokenizando o dataset"
    )
    print(f"Dataset processado em {time.time() - start_time:.2f} segundos.")
    
    # Configurar o data collator com padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100  # Usar -100 como valor de padding para os labels
    )
    
    # Criar diretório para os checkpoints
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Configurar os argumentos de treinamento
    training_args = Seq2SeqTrainingArguments(
        output_dir=CHECKPOINT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=SAVE_TOTAL_LIMIT,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=NUM_EPOCHS,
        predict_with_generate=True,
        fp16=False,  # Desativar fp16 para maior estabilidade
        max_grad_norm=1.0,  # Gradient clipping para prevenir explosão de gradientes
        gradient_accumulation_steps=4,  # Estabilizar o treinamento
        load_best_model_at_end=True,
        metric_for_best_model="exact_match",
        greater_is_better=True,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_strategy="steps",
        logging_steps=100,
        debug="underflow_overflow",
    )
    
    # Verificar se existe um checkpoint anterior
    last_checkpoint = get_last_checkpoint(CHECKPOINT_DIR)
    if last_checkpoint:
        print(f"\nRetomando treinamento a partir do checkpoint: {last_checkpoint}")
    else:
        print("\nIniciando treinamento do zero...")
    
    # Configurar o Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]
    )
    
    # Treinar o modelo (usando checkpoint se disponível)
    print("\nIniciando o treinamento...")
    try:
        # Implementar estratégia de tentativas múltiplas para o treinamento
        max_attempts = 3
        current_attempt = 1
        training_completed = False
        
        while current_attempt <= max_attempts and not training_completed:
            try:
                print(f"Tentativa de treinamento {current_attempt}/{max_attempts}")
                # Usa o parâmetro resume_from_checkpoint se existir um checkpoint anterior
                trainer.train(resume_from_checkpoint=last_checkpoint)
                training_completed = True
                print("Treinamento concluído com sucesso!")
            except RuntimeError as re:
                if "CUDA" in str(re) and current_attempt < max_attempts:
                    print(f"\nErro CUDA detectado: {str(re)}")
                    print(f"Tentando reiniciar o treinamento (tentativa {current_attempt+1}/{max_attempts})...")
                    # Limpar a memória CUDA
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # Reduzir o batch size para a próxima tentativa
                    new_batch_size = max(1, BATCH_SIZE // 2)
                    print(f"Reduzindo batch size para {new_batch_size}")
                    trainer.args.per_device_train_batch_size = new_batch_size
                    trainer.args.per_device_eval_batch_size = new_batch_size
                    # Aumentar gradient accumulation para compensar
                    trainer.args.gradient_accumulation_steps *= 2
                    print(f"Aumentando gradient_accumulation_steps para {trainer.args.gradient_accumulation_steps}")
                    # Atualizar o caminho do último checkpoint antes da próxima tentativa
                    last_checkpoint = get_last_checkpoint(CHECKPOINT_DIR)
                    current_attempt += 1
                else:
                    raise
    except KeyboardInterrupt:
        print("\nTreinamento interrompido pelo usuário.")
        print("O progresso foi salvo no último checkpoint.")
        training_completed = False
    except Exception as e:
        print(f"\nErro durante o treinamento: {str(e)}")
        print("O progresso foi salvo no último checkpoint.")
        training_completed = False
    
    if training_completed:
        # Avaliação final
        print("\nAvaliação final do modelo:")
        eval_results = trainer.evaluate()
        print(f"BLEU Score: {eval_results['eval_bleu']:.2f}")
        print(f"Exact Match: {eval_results['eval_exact_match']:.2f}")
        
        # Salvar o modelo final
        final_model_dir = os.path.join(OUTPUT_DIR, "final_model")
        os.makedirs(final_model_dir, exist_ok=True)
        print(f"\nSalvando o modelo final em {final_model_dir}")
        trainer.save_model(final_model_dir)
        print("Treinamento concluído!")
        
        # Teste rápido com pipeline
        print("\nTestando o modelo com um exemplo:")
        from transformers import pipeline
        
        exemplo_pt = dataset["test"][0]["pt"]
        print(f"Texto original: {exemplo_pt}")
        
        # Criar um pipeline de tradução
        tradutor = pipeline(
            "translation", 
            model=final_model_dir, 
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Traduzir o exemplo
        resultado = tradutor(f"translate Portuguese to LIBRAS: {exemplo_pt}", max_length=MAX_LENGTH)
        traducao = resultado[0]["translation_text"]
        
        # Converter para maiúsculo para manter o padrão LIBRAS
        traducao = traducao.upper()
        print(f"Tradução gerada: {traducao}")
        print(f"Referência: {dataset['test'][0]['lb'].upper()}")
        
        # Calcular BLEU manualmente para este exemplo
        from sacrebleu import corpus_bleu
        bleu_score = corpus_bleu([traducao], [[dataset['test'][0]['lb'].upper()]])
        print(f"BLEU Score para este exemplo: {bleu_score.score:.2f}")
        
        return eval_results
    else:
        print("\nTreinamento não foi concluído. Execute o script novamente para continuar.")
        return None

if __name__ == "__main__":
    # Configuração para habilitar depuração CUDA
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    # Criando diretórios necessários
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Carregar o tokenizador do modelo base
    print("Carregando tokenizador do modelo base...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_CHECKPOINT)
    
    # Garantir que o tokenizador tenha tokens de padding configurados corretamente
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Configurado token de padding para o tokenizador")
    
    # Adicionar tokens especiais ao tokenizador
    print("Adicionando tokens especiais ao tokenizador...")
    special_tokens = {"additional_special_tokens": ["ã", "õ", "í", "ú", "â", "ê", "ô"]}
    num_added_tokens = tokenizer.add_special_tokens(special_tokens)
    print(f"Adicionados {num_added_tokens} tokens especiais ao tokenizador")
    
    # Carregar modelo do checkpoint base
    print(f"Carregando modelo base {MODEL_CHECKPOINT}...")
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)
        print("Modelo base carregado com sucesso!")
        
        # Redimensionar os embeddings do modelo para acomodar os novos tokens
        print("Redimensionando embeddings do modelo...")
        model.resize_token_embeddings(len(tokenizer))
        print(f"Embeddings redimensionados para {len(tokenizer)} tokens")
    except Exception as e:
        print(f"Erro ao carregar modelo base: {str(e)}")
        exit(1)
    
    # Carregar métrica
    metric = evaluate.load("sacrebleu")
    
    # Executar a função principal
    main()