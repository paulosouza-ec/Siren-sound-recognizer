from base import *
import numpy as np

#espectral centrall indica onde o "centro de massa" do espectro de frequência de um sinal de áudio está localizado
#caracteriza "brilho" ou "tonalidade" do som

#O resultado do spectral centroid para cada audio sera salvo em um arquivo .csv

# Lista para armazenar os centros espectrais calculados
centros_espectrais = []

# Itera sobre todos os arquivos no diretório
for arquivo in os.listdir(diretorio_destino):
    if arquivo.endswith('.wav'):
        # Carrega o arquivo de áudio
        caminho_arquivo = os.path.join(diretorio_destino, arquivo)
        audio, _ = librosa.load(caminho_arquivo, sr=None)
        
        # Calcula o centroide espectral
        centro_espectral = librosa.feature.spectral_centroid(y=audio, sr=44100)[0]
        
        # Adiciona o centroide espectral à lista
        centros_espectrais.append((arquivo, np.mean(centro_espectral)))

# Exibe os centros espectrais calculados
for arquivo, centro_espectral in centros_espectrais:
    print(f"Arquivo: {arquivo}, Centroide Espectral: {centro_espectral}")

# Nome da pasta para salvar o arquivo
# Mude esse diretorio para alguma pasta que voce queira salvar o resultado do spectral centroid
pasta_destino = r'C:/Users/psgs2/Documents/uni/proj_final_sinais/centroid'

# Verifica e cria a pasta de destino se ela não existir
if not os.path.exists(pasta_destino):
    os.makedirs(pasta_destino)

# Nome do arquivo para salvar os resultados
arquivo_resultados = os.path.join(pasta_destino, 'resultados_centroides.csv')

# Salva os resultados em um arquivo CSV
np.savetxt(arquivo_resultados, centros_espectrais, delimiter=',', fmt='%s')