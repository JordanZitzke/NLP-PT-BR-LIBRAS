# NLP-PT-BR-LIBRAS

# 🇧🇷 Portuguese to LIBRAS Translation with Speech-to-Text Integration

## 📌 Descrição do Projeto
Este repositório contém a implementação de um sistema ponta-a-ponta para transcrição de áudio e tradução de sentenças do Português para a glosa da Língua Brasileira de Sinais (LIBRAS). O projeto utiliza modelos baseados na arquitetura **Transformer** para Processamento de Linguagem Natural (NLP) e o framework Faster Whisper para reconhecimento de fala, com o objetivo de facilitar a acessibilidade para pessoas surdas.

Este código reflete a pesquisa detalhada no artigo: *"Improving Translation from Portuguese to Brazilian Sign Language with Speech-to-Text Integration through Natural Language Processing"*.

## 🏗️ Arquitetura e Tecnologias
O sistema é composto por duas frentes principais:
1. **Speech-to-Text (ASR):** Reconhecimento de fala utilizando o `faster-whisper` (`transcrever.py`).
2. **Tradução Neural (NMT):** Fine-tuning do modelo `T5-small` usando a biblioteca Hugging Face `transformers` para conversão PT -> LIBRAS (`treinamento_mod.py` e `Teste_sentenca.py`).

**Stack Tecnológico:**
* Python 3.x
* PyTorch
* Hugging Face Transformers
* Faster Whisper
* NLTK / Datasets

## 📂 Estrutura do Repositório
* `transcrever.py`: Script para transcrição de áudio em tempo real ou via arquivo de áudio.
* `treinamento_mod.py`: Pipeline de treinamento e fine-tuning do modelo T5 para tradução.
* `Teste_sentenca.py`: Script de inferência que carrega o modelo treinado para traduzir sentenças interativamente.
* `dataset_minusculo.json`: Amostra do dataset com os pares de tradução (PT/LIBRAS).

## 🚀 Instalação e Configuração

1. Clone o repositório:
```bash
git clone [https://github.com/seu-usuario/nome-do-repositorio.git](https://github.com/seu-usuario/nome-do-repositorio.git)
cd nome-do-repositorio
```
2. Crie e ative um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```
3. Instale as dependências:
```bash
pip install torch transformers datasets evaluate faster-whisper sounddevice numpy nltk
```

## ⚙️ Uso

1. Transcrição de Áudio:
```bash
python transcrever.py --model base --device cpu
```
2. Treinamento do Modelo
```bash
python treinamento_mod.py
```
3. Inferência / Tradução de Textos:
```bash
python Teste_sentenca.py
```
