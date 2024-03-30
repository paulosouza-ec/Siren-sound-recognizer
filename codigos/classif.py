import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# Diretórios das pastas contendo os espectrograma, as imagens de loudness e o espectro centroid

#Altere o diretorio para cada uma das pastas onde voce colocou o arquivo de, respectivamente, espectogramas, loudness e o arquivo .csv espectral centroid
spectrogram_dir = r'C:/Users/psgs2/Documents/uni/proj_final_sinais/pasta_02'
loudness_dir = r'C:/Users/psgs2/Documents/uni/proj_final_sinais/loudness'
spectral_centroid_file = r'C:/Users/psgs2/Documents/uni/proj_final_sinais/centroid/resultados_centroides.csv'

centroides_df = pd.read_csv(spectral_centroid_file, header=None)
centroides_df.columns = ['filename', 'spectral_centroid']

# Criar um DataFrame para armazenar todas as características combinadas
combined_features = []

# Listar os arquivos nos diretórios de espectrograma e loudness
spectrogram_files = os.listdir(spectrogram_dir)
loudness_files = os.listdir(loudness_dir)

assert len(spectrogram_files) == len(loudness_files) == len(centroides_df), "The number of files in the directories and centroid dataframe must match."

for index, row in centroides_df.iterrows():
    filename = row['filename'].replace('.wav', '')  # Remover extensão .wav para correspondência com arquivos de imagem

    # Carregar e processar o espectrograma
    spectrogram_path = os.path.join(spectrogram_dir, filename + '_espectrograma.png')
    spectrogram_image = cv2.imread(spectrogram_path, cv2.IMREAD_GRAYSCALE)  # Carregar como escala de cinza
    spectrogram_arr = spectrogram_image.flatten()  # Transformar em um vetor 1D

    # Carregar e processar a imagem de loudness
    loudness_path = os.path.join(loudness_dir, filename + '_energia.png')
    loudness_image = cv2.imread(loudness_path, cv2.IMREAD_GRAYSCALE)  # Carregar como escala de cinza
    loudness_arr = loudness_image.flatten()  # Transformar em um vetor 1D

    # Obter o valor do centroide espectral
    spectral_centroid_value = row['spectral_centroid']

    # Combinar as características em um único vetor
    combined_vector = np.hstack((spectrogram_arr, loudness_arr, spectral_centroid_value))

    # Adicionar ao DataFrame de características combinadas
    combined_features.append(combined_vector)

combined_features_array = np.array(combined_features)

# Certificar-se de que as dimensões dos vetores de características são compatíveis
assert combined_features_array.shape[1] == (len(spectrogram_arr) + len(loudness_arr) + 1), "As dimensões dos vetores de características não são compatíveis."

# Agora, `combined_features_array` contém todas as características combinadas para cada amostra de áudio.
# Cada linha representa uma amostra de áudio e cada coluna representa uma característica.

print("Shape of combined features array:", combined_features_array.shape)

# Separar os dados em features (X) e rótulos (Y)
X = combined_features_array  # Vetores de características combinadas

scaler = StandardScaler()
X_scaled = scaler.fit_transform(combined_features_array)

#Quantidade de grupos
num_clusters = 4 

#inicializacao do modelo kmeans
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X_scaled)  #dados normalizados

#obter rotulo p/ cada audio
cluster_labels = kmeans.labels_

print("Cluster labels for all examples:")
print(cluster_labels)

def atribuir_rotulos_cluster(cluster_labels):
    #mapeando rotulos de clusters p/ classes
    mapping = {0: "ambulância", 1: "bombeiro", 2: "polícia", 3: "alarme"}
    rotulos_classe = [mapping[label] for label in cluster_labels]
    return rotulos_classe

#rotulos de fato
rotulos_classe = atribuir_rotulos_cluster(cluster_labels)
print("rotulos classe: ", rotulos_classe)


# Nos escolhemos dividir os dados de forma randomica (random_state = 42), logo como a base de dados eh pequena pode ser que nao consiga fazer
# a escolha dos melhores arquivos de treinamento para conseguir detectar os padroes, mas ainda assim consegue classificar baseado nos padroes.

X_train, X_test, Y_train, Y_test = train_test_split(combined_features_array, rotulos_classe, test_size=0.32, random_state=42)

#Inicializar e treinar o modelo SVM
modelo = SVC(kernel='linear', random_state=42)
modelo.fit(X_train, Y_train)

#prever os rótulos para os dados de teste
Y_pred = modelo.predict(X_test)

#agr precisao
precisao = accuracy_score(Y_test, Y_pred)
print("Precisão do modelo:", precisao)


rotulos_unicos = np.unique(Y_test)

#obter o relatório de classificação como uma string
report_string = classification_report(Y_test, Y_pred, labels=rotulos_unicos)

print("Relatório de classificação:")
print(report_string)

matriz_confusao = confusion_matrix(Y_test, Y_pred)

#matriz de confusão como um heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_confusao, annot=True, cmap='Blues', fmt='g', xticklabels=rotulos_unicos, yticklabels=rotulos_unicos)
plt.xlabel('Rótulos Previstos')
plt.ylabel('Rótulos Verdadeiros')
plt.title('Matriz de Confusão')
plt.show()

#----------------

# Convertendo as classes em rótulos binários
classes = ["ambulância", "bombeiro", "polícia", "alarme"]
Y_test_binary = label_binarize(Y_test, classes=classes)

# Calculando as probabilidades de classe para os dados de teste
Y_prob = modelo.decision_function(X_test)

# Calculando a Curva ROC 
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(classes)):
    fpr[i], tpr[i], _ = roc_curve(Y_test_binary[:, i], Y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plotando a Curva ROC
plt.figure()
for i in range(len(classes)):
    plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve of class {} (area = {:0.2f})'.format(classes[i], roc_auc[i]))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()

# Dois graficos serao gerados (matriz de confusao e curva ROC).