import wave
import os
import shutil
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import librosa

diretorio_origem = r'C:/Users/psgs2/Documents/uni/proj_final_sinais/audio' # Deve-se mudar esse diretorio! Esse diretorio deve conter os audios do dataset
diretorio_destino = r'C:/Users/psgs2/Documents/uni/proj_final_sinais/siren_audio'
#Deve-se mudar o diretorio_destino para alguma pasta que vocês queiram que o audio das sirenes fiquem armazenados.


# Em primeiro lugar, no codigo abaixo, a ideia é capturar apenas os audios de sirene da base de dados (pasta audio). Sao 40 audios.

# Criar o diretório de destino se ele não existir
if not os.path.exists(diretorio_destino):
    os.makedirs(diretorio_destino)

# Lista dos arquivos de sirene selecionados
arquivos_selecionados = [arquivo for arquivo in os.listdir(diretorio_origem) if arquivo.endswith('.wav') and '-42.wav' in arquivo]

# Cópia os arquivos selecionados para o diretório de destino
for arquivo in arquivos_selecionados:
    caminho_origem = os.path.join(diretorio_origem, arquivo)
    caminho_destino = os.path.join(diretorio_destino, arquivo)
    shutil.copy2(caminho_origem, caminho_destino)

print("Arquivos de audio da sirene foram copiados com sucesso para:", diretorio_destino)


# A ideia desse codigo é colocar os audios de sirene dentro de um vetor de tupla chamado 'audios'
# Aqui os audios sao organizados em cada elemento da lista como : (nome do arquivo, endereco do audio na memoria)

# Lista para armazenar os arquivos de áudio abertos
audios = []
# Itera sobre todos os arquivos no diretório
for arquivo in os.listdir(diretorio_destino):
    if arquivo.endswith('.wav'):
        # Abre o arquivo de áudio
        caminho_arquivo = os.path.join(diretorio_destino, arquivo)
        audio = wave.open(caminho_arquivo, 'rb')
        audios.append((arquivo, audio))  # Adiciona tupla (nome do arquivo, objeto de áudio [endereco])

