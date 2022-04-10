#BRUNO EDUARDO FARIAS -  GRR20186715
#CDD 2021-2

import csv, sys, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime;
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def load_data(nome_arquivo):
	input_data = pd.read_csv(nome_arquivo)

	row, col = input_data.shape
	print(f'Linhas: {row} Colunas: {col}') 
	print(input_data.head(10))

	#salva a classificação
	y = input_data['Class'].tolist()

	#remove a classificação (col Class)
	input_data = input_data.drop('Class', axis = 1)

	#remove a classificação (col Class) e o atributo Time
	#input_data = input_data.drop(['Time', 'Class'], axis = 1)

	print('Após drops: ')
	print(input_data.head(10))	

	input_data_scaled = input_data.copy()

	#normalizando dados
	print('Normalizando...')
	input_data_scaled[input_data_scaled.columns] = StandardScaler().fit_transform(input_data_scaled)

	print(input_data_scaled.describe())

	return input_data_scaled, y

def pca_embeddings(data_scaled):
	pca_2 = PCA(n_components=2)
	pca_2_result = pca_2.fit_transform(data_scaled)

	#imprime dados sobre o PCA
	#dataset_pca = pd.DataFrame(abs(pca_2.components_), columns=data_scaled.columns, index=['PC_1', 'PC_2'])
	#print('\n\n', dataset_pca)
    
	#print('Componente principal 1:\n', (dataset_pca[dataset_pca > 0.3].iloc[0]).dropna())
	#print('\n\nComponente principal 2:\n', (dataset_pca[dataset_pca > 0.3].iloc[1]).dropna())

	return pca_2_result, pca_2

# def plotar_resultados(pca_result, label, centroids_pca):
# 	x = pca_result[:, 0]
# 	y = pca_result[:, 1]

# 	plt.scatter(x, y, c=label, alpha=0.5, s=100)
# 	plt.title('Frauds clusters')
# 	plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='X', s=200, linewidths=1.5,
#                 color='red', edgecolors="black", lw=1.5)
# 	plt.show()

def plotar_resultados(pca_result, label, centroids_pca, kmeans):
	h = 0.02

	x_min, x_max = pca_result[:, 0].min() - 1, pca_result[:, 0].max() + 1
	y_min, y_max = pca_result[:, 1].min() - 1, pca_result[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

	Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

	Z = Z.reshape(xx.shape)
	plt.figure(1)
	plt.clf()
	plt.imshow(
		Z,
		interpolation="nearest",
		extent=(xx.min(), xx.max(), yy.min(), yy.max()),
		cmap=plt.cm.Paired,
		aspect="auto",
		origin="lower",
	)

	plt.plot(pca_result[:, 0], pca_result[:, 1], "k.", markersize=2)
	centroids = kmeans.cluster_centers_
	plt.scatter(
		centroids_pca[:, 0],
		centroids_pca[:, 1],
		marker="x",
		s=169,
		linewidths=3,
		color="w",
		zorder=10,
	)

	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)
	plt.xticks(())
	plt.yticks(())
	plt.show()

def exporacao():
	print('Iniciando exploração do dataset...')

	#abrindo dataset que veio pelo sys.argv[1]
	nome_arquivo = sys.argv[1]

	print('1. Lendo dados %s...' % (nome_arquivo))
	data_scaled, y = load_data(nome_arquivo)

	data_scaled = data_scaled.to_numpy()

	print('2. Reduzindo via PCA...')
	pca_result, pca_2 = pca_embeddings(data_scaled)

	print('3. Fazendo KMeans...')
	# fitting KMeans
	kmeans = KMeans(n_clusters=2)
	#kmeans.fit(data_scaled)
	kmeans.fit(pca_result)
	centroids = kmeans.cluster_centers_
	#centroids_pca = pca_2.transform(centroids)

	print('4. Visualizando dados...')
	#plotar_resultados(pca_result, kmeans.labels_, centroids_pca)
	plotar_resultados(pca_result, kmeans.labels_, centroids, kmeans)

if __name__ == "__main__":
	exporacao()