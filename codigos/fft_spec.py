from base import * 
from scipy.io import wavfile
from scipy.signal import spectrogram


#A ideia eh atraves da FFT, retornar o spectograma (frequencia ao longo do tempo) para cada cada audio.
diretorio_destino_plots = r'C:/Users/psgs2/Documents/uni/proj_final_sinais/spectogram' #Mude esse diretorio
#Mude esse diretorio para alguma pasta que voce quer armazenar o espectograma de cada audio em png

# Criar o diretório de destino se ele não existir
if not os.path.exists(diretorio_destino_plots):
    os.makedirs(diretorio_destino_plots)

# Itera sobre todos os arquivos no diretório
for arquivo in os.listdir(diretorio_destino):
    if arquivo.endswith('.wav'):
        # Abre o arquivo de áudio
        caminho_arquivo = os.path.join(diretorio_destino, arquivo)
        taxa_amostragem, dados_audio = wav.read(caminho_arquivo)

        # Calcula a Transformada de Fourier de Curto Período (STFT)
        f, t, espectrograma = spectrogram(dados_audio, fs=taxa_amostragem, nperseg=1024, noverlap=512)

        #Exibicao do espectograma
        plt.figure()
        plt.pcolormesh(t, f, 10 * np.log10(espectrograma), shading='auto')
        plt.colorbar(label='Log(Energia)')

        # Configurações para uma boa plotagem
        plt.xlabel('Tempo (s)')
        plt.ylabel('Frequência (Hz)')
        plt.title('Espectrograma do Áudio: ' + arquivo)

        # Salva o gráfico no diretório de destino
        caminho_destino_plot = os.path.join(diretorio_destino_plots, arquivo.replace('.wav', '_espectrograma.png'))
        plt.savefig(caminho_destino_plot, format='png')

        plt.close()

        print("Espectrograma calculado e salvo para:", arquivo)