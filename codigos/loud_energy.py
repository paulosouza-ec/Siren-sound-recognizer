from base import *
import os
import numpy as np


# A ideia do codigo eh retornar uma imagem da energia do audio ao longo de suas amostras.
# Lista para armazenar a Loudness calculada. 
#Essa parte demora um pouco mais p/ gerar todas as imagens. Nao dura mais de 30 segundos.
#Aparecera uma msg que os arquivos Loudness foram salvos.
loudness_list = []

# Itera sobre todos os arquivos no diretório
for arquivo in os.listdir(diretorio_destino):
    if arquivo.endswith('.wav'):
        # Carrega o arquivo de áudio
        caminho_arquivo = os.path.join(diretorio_destino, arquivo)
        audio, taxa_amostragem = librosa.load(caminho_arquivo, sr=None)
        
        # Calcula a intensidade sonora (loudness)
        loudness = np.sqrt(np.mean(audio**2))

        # Adiciona a Loudness a lista
        loudness_list.append((arquivo, loudness))

        # Salva a imagem da energia do sinal
        plt.figure()
        plt.plot(audio)
        plt.xlabel('Amostras')
        plt.ylabel('Amplitude')
        plt.title('Energia do Sinal')
        
        # Criar o diretório de destino se ele não existir
        # Mude esse diretorio "pasta_destino" para alguma pasta que voce queira salvar as imagens do loudness para cada audio
        pasta_destino_imagem = r'C:/Users/psgs2/Documents/uni/proj_final_sinais/loudness'
        if not os.path.exists(pasta_destino_imagem):
            os.makedirs(pasta_destino_imagem)

        caminho_destino_imagem = os.path.join(pasta_destino_imagem, arquivo.replace('.wav', '_energia.png'))
        plt.savefig(caminho_destino_imagem, format='png')
        plt.close()
        print(f"Os arquivos  loudness estao sendo gerados.... aguarde....")

print(f"Os resultados de Loudness foram salvos na pasta especificada!")



