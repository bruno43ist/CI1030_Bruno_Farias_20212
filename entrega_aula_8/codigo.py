#BRUNO EDUARDO FARIAS -  GRR20186715
#CDD 2021-2

import csv, sys, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def exporacao():
	print('Iniciando exploração do dataset...')

	#abrindo dataset que veio pelo sys.argv[1]
	nome_arquivo = sys.argv[1]

	print('Lendo arquivo %s...' % (nome_arquivo))

	try:
		#abre arquivo
		with open(nome_arquivo, 'r') as arquivo:
			try:
				#le as linhas do csv
				reader = csv.reader(arquivo)

				quantidadeDeLinhas = sum(1 for linha in reader)
				quantidadeDeLinhas -= 1 #retira cabeçalho = quantidade de amostras

				print('Quantidade de amostras: %d' % (quantidadeDeLinhas))

				#volta pro início do arquivo
				arquivo.seek(0,0)

				print('Cabecalho:')
				rowCabecalho = next(reader)
				print(rowCabecalho)

				#verifica se quer gerar scatterplot
				imprimir = input('Deseja imprimir as amostras? Quantidade: %d (S/n)' % (quantidadeDeLinhas))
				print(imprimir)
			
				contFraudes = 0
				for linha in reader:
					if imprimir == 'S' or imprimir == 's':
						print(linha)
					if linha[30] == '1':
						#somatório para total de fraudes
						contFraudes += 1

				#calcula quantidade de amostras sem fraudes
				qtdAmostrasSemFraude = quantidadeDeLinhas - contFraudes

				print('Quantidade de Amostras COM Fraudes: %d' % (contFraudes))
				print('Quantidade de Amostras SEM Fraudes: %d' % (qtdAmostrasSemFraude))

				print('Gerando gráfico...')

				#gera gráfico de barras
				nomes = ['Com Fraude', 'Sem Fraude']
				valores = [contFraudes, qtdAmostrasSemFraude]

				bars = plt.bar(nomes, valores)
				for bar in bars:
					yval = bar.get_height()
					plt.text(bar.get_x() + 0.28, yval + .005, yval)
				plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
				plt.title('Amostras por classe')
				plt.xlabel('Classes')
				plt.ylabel('Quantidade')
				plt.grid(color='b', linestyle='--', linewidth=1, axis='y')
				plt.show()

			except csv.Error as e:
					sys.exit('arquivo %s, linha %d: %s' % (nome_arquivo, reader.line_num, e))
	except FileNotFoundError as e:
		sys.exit('Arquivo \'%s\' não encontrado!' % (nome_arquivo))

if __name__ == "__main__":
	exporacao()
