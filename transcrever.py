import sounddevice as sd
import numpy as np
import wave
import os
import tempfile
import time
import queue
from faster_whisper import WhisperModel
import threading
import nltk
from nltk.metrics.distance import edit_distance

# Baixar recursos do NLTK necessários para processamento de texto
nltk.download('punkt', quiet=True)

class AudioProcessor:
    def __init__(self, model_size="base", device="cpu", compute_type="int8", segment_length=3):
        """
        Inicializa o processador de áudio.
        """
        self.segment_length = segment_length
        self.sample_rate = 16000
        self.channels = 1
        self.dtype = np.int16
        
        # Inicializar modelo Whisper
        print("Inicializando modelo Faster Whisper...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("Modelo inicializado com sucesso!")
        
        # Filas para comunicação entre threads
        self.audio_queue = queue.Queue()
        self.text_queue = queue.Queue()
        
        # Flag para controle
        self.running = False
        
        # Histórico para correção de palavras cortadas
        self.last_transcript = ""
        
        # Lista para armazenar todas as transcrições
        self.all_transcripts = []
    
    def start_processing(self):
        """Inicia as threads de captura e transcrição de áudio"""
        self.running = True
        
        # Iniciar thread de captura de áudio
        self.recorder_thread = threading.Thread(target=self._record_audio)
        self.recorder_thread.daemon = True
        self.recorder_thread.start()
        
        # Iniciar thread de transcrição
        self.transcriber_thread = threading.Thread(target=self._transcribe_audio)
        self.transcriber_thread.daemon = True
        self.transcriber_thread.start()
        
        # Iniciar thread para processar texto
        self.text_processor_thread = threading.Thread(target=self._process_text)
        self.text_processor_thread.daemon = True
        self.text_processor_thread.start()
    
    def stop_processing(self):
        """Para todas as threads"""
        self.running = False
        
        # Aguardar finalização das threads (com timeout)
        if hasattr(self, 'recorder_thread') and self.recorder_thread.is_alive():
            self.recorder_thread.join(timeout=1)
        
        if hasattr(self, 'transcriber_thread') and self.transcriber_thread.is_alive():
            self.transcriber_thread.join(timeout=1)
        
        if hasattr(self, 'text_processor_thread') and self.text_processor_thread.is_alive():
            self.text_processor_thread.join(timeout=1)
        
        print("\nTranscrição completa:")
        print(" ".join(self.all_transcripts))
    
    def _record_audio(self):
        """Captura continuamente segmentos de áudio e os coloca na fila."""
        try:
            print("Iniciando gravação de áudio...")
            
            while self.running:
                # Gravar segmento de áudio
                print(f"Capturando segmento de {self.segment_length}s...")
                audio_data = sd.rec(
                    int(self.sample_rate * self.segment_length),
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype=self.dtype
                )
                sd.wait()  # Aguardar término da gravação
                
                # Colocar dados de áudio na fila para processamento
                if self.running:
                    self.audio_queue.put(audio_data)
                    print("Segmento capturado e enviado para processamento")
                
        except Exception as e:
            print(f"Erro na gravação de áudio: {e}")
            self.running = False
    
    def _transcribe_audio(self):
        """Processa segmentos de áudio da fila e os transcreve."""
        print("Iniciando processo de transcrição...")
        
        while self.running:
            try:
                # Obter segmento de áudio da fila com timeout para verificar o flag running
                try:
                    audio_data = self.audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                print("Processando segmento de áudio...")
                
                # Salvar dados temporariamente para processamento
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_filename = temp_file.name
                
                # Salvar áudio no arquivo temporário
                with wave.open(temp_filename, 'wb') as wf:
                    wf.setnchannels(self.channels)
                    wf.setsampwidth(2)  # 16 bits = 2 bytes
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(audio_data.tobytes())
                
                # Transcrever áudio
                print("Transcrevendo áudio...")
                segments, _ = self.model.transcribe(temp_filename, language="pt")
                transcript = " ".join([segment.text for segment in segments]).strip()
                
                # Enviar transcrição para processamento
                if transcript:
                    self.text_queue.put(transcript)
                    print(f"Transcrição: '{transcript}'")
                else:
                    print("Nenhuma fala detectada neste segmento")
                
                # Remover arquivo temporário
                os.unlink(temp_filename)
                
            except Exception as e:
                if self.running:  # Ignorar erros durante o encerramento
                    print(f"Erro na transcrição: {e}")
    
    def _process_text(self):
        """Processa transcrições, corrigindo palavras cortadas."""
        print("Iniciando processamento de texto...")
        
        while self.running:
            try:
                # Obter transcrição da fila com timeout para verificar o flag running
                try:
                    transcript = self.text_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Corrigir palavras cortadas
                corrected_transcript = self._fix_cut_words(self.last_transcript, transcript)
                
                # Exibir a transcrição
                print(f"Texto final: {corrected_transcript}")
                
                # Armazenar para exibição final
                self.all_transcripts.append(corrected_transcript)
                
                # Atualizar última transcrição
                self.last_transcript = transcript
                
            except Exception as e:
                if self.running:
                    print(f"Erro no processamento de texto: {e}")
    
    def _fix_cut_words(self, prev_text, current_text):
        """Corrige palavras cortadas entre segmentos."""
        if not prev_text or not current_text:
            return current_text
        
        # Pegar a última palavra do texto anterior
        prev_words = prev_text.split()
        if not prev_words:
            return current_text
        
        last_word = prev_words[-1].lower()
        
        # Pegar a primeira palavra do texto atual
        current_words = current_text.split()
        if not current_words:
            return current_text
        
        first_word = current_words[0].lower()
        
        # Verificar se a primeira palavra do texto atual pode ser continuação da última do anterior
        if len(first_word) >= 3 and (first_word in last_word or last_word in first_word):
            # Se as palavras são muito similares, provavelmente são a mesma
            if edit_distance(last_word, first_word) <= min(len(last_word), len(first_word)) / 2:
                # Usar a palavra mais longa
                if len(last_word) > len(first_word):
                    merged_word = last_word
                else:
                    merged_word = first_word
                
                return merged_word + " " + " ".join(current_words[1:])
        
        # Se não houver correção necessária, retornar o texto atual
        return current_text


def process_audio_file(file_path, model_size="base", device="cpu", compute_type="int8"):
    """Processa um arquivo de áudio em vez de capturar ao vivo."""
    print(f"Processando arquivo: {file_path}")
    
    # Inicializar o modelo Whisper
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    
    # Transcrever o arquivo completo
    segments, info = model.transcribe(file_path, language="pt")
    
    print("\nTranscrição completa:")
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")


def setup_audio_device():
    """Configura o dispositivo de áudio para captura."""
    print("\n=== Verificando dispositivos de áudio disponíveis... ===")
    try:
        devices = sd.query_devices()
        print(f"Total de dispositivos encontrados: {len(devices)}")
        
        input_devices = []
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"{i}: {device['name']} (entradas: {device['max_input_channels']})")
                input_devices.append(i)
        
        if not input_devices:
            print("ERRO: Nenhum dispositivo de entrada de áudio encontrado!")
            print("Isso pode ocorrer porque você está em um ambiente virtual.")
            return None
        
        device_id = None
        if len(input_devices) > 1:
            print("\nMúltiplos dispositivos de entrada encontrados. Por favor, selecione um:")
            while device_id is None:
                try:
                    choice = int(input("Digite o número do dispositivo de entrada a usar: "))
                    if choice in input_devices:
                        device_id = choice
                    else:
                        print(f"Escolha inválida. Digite um número entre: {input_devices}")
                except ValueError:
                    print("Por favor, digite um número válido.")
        else:
            device_id = input_devices[0]
            print(f"Usando o único dispositivo disponível: {devices[device_id]['name']}")
        
        # Configurar o dispositivo de entrada
        sd.default.device = device_id
        
        # Testar gravação breve
        print("Testando dispositivo de áudio (1 segundo)...")
        test_audio = sd.rec(
            int(16000 * 1),
            samplerate=16000,
            channels=1,
            dtype=np.int16
        )
        sd.wait()
        print("Teste concluído com sucesso!")
        
        return device_id
        
    except Exception as e:
        print(f"Erro ao configurar áudio: {e}")
        return None


def main():
    """Função principal do programa."""
    print("=== Sistema de Transcrição Contínua de Fala ===")
    
    # Configurar dispositivo de áudio
    device_id = setup_audio_device()
    if device_id is None:
        print("Não foi possível configurar o dispositivo de áudio.")
        return
    
    # Inicializar processador de áudio
    processor = AudioProcessor(model_size="base", device="cpu", compute_type="int8")
    
    try:
        # Iniciar processamento
        print("Iniciando transcrição em tempo real... (Pressione Ctrl+C para parar)")
        processor.start_processing()
        
        # Manter o programa em execução até Ctrl+C
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nInterrompendo...")
    finally:
        # Parar processamento e exibir resultados
        processor.stop_processing()


# Ponto de entrada principal
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Transcrição de fala para texto")
    parser.add_argument("--file", type=str, help="Caminho para o arquivo de áudio a ser processado")
    parser.add_argument("--model", type=str, default="base", 
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Tamanho do modelo Whisper")
    parser.add_argument("--device", type=str, default="cpu", 
                        choices=["cpu", "cuda"], 
                        help="Dispositivo para processamento")
    
    args = parser.parse_args()
    
    print("Sistema de transcrição de fala para texto iniciando...")
    
    if args.file:
        # Modo de processamento de arquivo
        process_audio_file(args.file, args.model, args.device)
    else:
        # Modo de captura de áudio ao vivo
        main()