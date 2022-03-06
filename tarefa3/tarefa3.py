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

				print('Classe das amostras: ')
				rowCabecalho = next(reader)
				print(rowCabecalho)
			
				print('No caso do dataset de Fraudes em Cartão de Crédito:')
				print('Time é a quantidade de tempo em segundos que passou desde a primeira transação,')
				print('V1 a V28: componentes obtidos com Análise de Componentes Principais (PCA - Principal Component Analysis) a respeito das transações. (O significado desses campos não é revelado por motivos de segurança.)')
				print('Amount é a quantia de dinheiro envolvida na transação')
				print('e Class indica se houve fraude nessa transação (1 se houve, 0 caso contrário).')
				
				print('Analisando fraudes...')

				contFraudes = 0
				listaAmountFraudes = []
				listaTimeFraudes = []
				listaClass = []
				
				#percorre todas as amostras
				for linha in reader:

					#adiciona a coluna 'Class' em uma lista (indica se a mostra x é fraude ou não)
					listaClass += [int(linha[30])]

					#se 'Class' = 1
					if linha[30] == '1':
						#somatório para total de fraudes
						contFraudes += 1
						#adiciona o valor na lista de valores das fraudes
						listaAmountFraudes += [float(linha[29])]
						#adiciona o tempo que passou desde a primeira transação até o momento dessa fraude
						listaTimeFraudes += [int(linha[0])]

				#calcula quantidade de amostras sem fraudes
				qtdAmostrasSemFraude = quantidadeDeLinhas - contFraudes

				print('Quantidade de Amostras COM Fraudes: %d' % (contFraudes))
				print('Quantidade de Amostras SEM Fraudes: %d' % (qtdAmostrasSemFraude))

				#volta pro início do arquivo
				arquivo.seek(0,0)

				print('Fazendo leitura com pandas...')
				data = pd.read_csv(nome_arquivo)
				#imprime estatísticas descritivas sobre o csv, como média, mediana, desvio padrão, etc...
				print(data.describe().T)

				print('Onde: mean é a média dos valores, std é o desvio padrão dos valores, min e max são os valores mínimos e máximos (respectivamente)')
				print('e \'25%\', \'50%\' e \'75%\' são os quartis.')

				print('De maneira geral as amostras são distinguíveis, pois é possível analisar as transações que são fraudulentas,')
				print('o valor dessas fraudes, o timestamp que ocorreram as fraudes, etc...')
				print('Porém, como os significado dos valores V1 a V28 não são revelados, não é possível fazer análises a respeito')
				print('das variações dos valores dessas colunas.')

				#verifica se quer gerar scatterplot
				gerar = input('Deseja gerar os scatterplot\'s? (S/n)')
				print(gerar)

				if gerar == 'S' or gerar == 's':
					print('Gerando scatterplot...')

					#gera gráfico de quantidade de amostras com fraude
					plt.plot(range(quantidadeDeLinhas),listaClass,'ro')
					plt.ylabel('Amostras')
					plt.ylabel('0 = sem fraude, 1 = fraude')
					plt.title('Distribuição das fraudes ao longo das amostras')
					plt.show()

					#gera gráfico de quantidedade de fraudes por faixa de valores
					#50, 100, 200, 400, 600, 800, 1000
					nomes_x = ['0-50', '50-100', '100-200', '200-400', '400-600', '600-800', '800-1000', '+1000']
					listaContagemAmountFraudes = [0,0,0,0,0,0,0,0]

					for amount in listaAmountFraudes:
						if amount > 1000:
							listaContagemAmountFraudes[7] += 1
						elif amount > 800:
							listaContagemAmountFraudes[6] += 1
						elif amount > 600:
							listaContagemAmountFraudes[5] += 1
						elif amount > 400:
							listaContagemAmountFraudes[4] += 1
						elif amount > 200:
							listaContagemAmountFraudes[3] += 1
						elif amount > 100:
							listaContagemAmountFraudes[2] += 1
						elif amount > 50:
							listaContagemAmountFraudes[1] += 1
						else:
							listaContagemAmountFraudes[0] += 1

					plt.bar(nomes_x, listaContagemAmountFraudes)
					plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
					plt.title('Valores x Fraudes')
					plt.xlabel('Faixas de Valores')
					plt.ylabel('Quantidade de ocorrências de fraudes')
					plt.grid(color='b', linestyle='--', linewidth=1, axis='y')
					plt.show()

					#gera gráfico de quantidedade de fraudes por faixa de tempo (segundos)
					#50, 100, 200, 400, 600, 800, 1000
					nomes_x = ['0-10000', '10000-20000', '20000-50000', '50000-100000', '100000-150000', '+150000']
					listaContagemTimeFraudes = [0,0,0,0,0,0]

					for time in listaTimeFraudes:
						if time > 150000:
							listaContagemTimeFraudes[5] += 1
						elif time > 100000:
							listaContagemTimeFraudes[4] += 1
						elif time > 50000:
							listaContagemTimeFraudes[3] += 1
						elif time > 20000:
							listaContagemTimeFraudes[2] += 1
						elif time > 10000:
							listaContagemTimeFraudes[1] += 1
						else:
							listaContagemTimeFraudes[0] += 1

					plt.bar(nomes_x, listaContagemTimeFraudes)
					plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
					plt.title('Time x Fraudes')
					plt.xlabel('Faixas de Tempo (segundos) após a ocorrência da primeira transação')
					plt.ylabel('Quantidade de ocorrências de fraudes')
					plt.grid(color='b', linestyle='--', linewidth=1, axis='y')
					plt.show()

				print('Final da exploração do dataset!')
			except csv.Error as e:
					sys.exit('arquivo %s, linha %d: %s' % (nome_arquivo, reader.line_num, e))
	except FileNotFoundError as e:
		sys.exit('Arquivo \'%s\' não encontrado!' % (nome_arquivo))

if __name__ == "__main__":
	exporacao() 