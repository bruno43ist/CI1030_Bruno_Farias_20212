# CI1030_Bruno_Farias_20212
Repositório criado para as tarefas da disciplina Ciência de Dados para Segurança - UFPR - 2021-2
Bruno E. Farias - GRR20186715

<p align="center">
<img src="http://img.shields.io/static/v1?label=STATUS&message=EM%20DESENVOLVIMENTO&color=GREEN&style=for-the-badge"/>
</p>

### Tarefas
:point_right: [Tarefa 1](#tarefa-1)

:point_right: [Tarefa 2](#tarefa-2)

:point_right: [Tarefa 3](#tarefa-3)

:point_right: [Entrega da aula 8](#entrega-da-aula-8)

:point_right: [Tarefa 4](#tarefa-4)

:new: [Classificação / Tarefa Final](#classificação-Tarefa-Final)


## Tarefa 1 
Contador de pacotes/sessões de um arquivo .pcap.

:mag_right: [Ver arquivo conta_sessoes.py](tarefa1/conta_sessoes.py)

## Tarefa 2
.txt entregue na tarefa 2

:mag_right: [Ver arquivo Bruno_Farias.txt](tarefa2/Bruno_Farias.txt)

## Tarefa 3
:credit_card: Repositório: Fraudes em Cartões de Crédito https://www.kaggle.com/samkirkiles/credit-card-fraud/data

Dados:

:receipt: Quantidades de Amostras: 284.807

:arrow_right: Amostras COM Fraude: 492

:arrow_right: Amostras SEM Fraude: 284315


Classes:

:clock1230: Time: quantidade de tempo em segundos que passou desde a primeira transação.

:question: V1-V28: componentes obtidos com Análise de Componentes Principais (PCA - Principal Component Analysis) a respeito das transações. (O significado desses campos não é revelado por motivos de segurança).

:euro: Amount é a quantia de dinheiro envolvida na transação.

:warning: Class indica se houve fraude nessa transação (1 se houve, 0 caso contrário).


Scatterplot's:

:one:) Fraudes por Amostras: indica se houve fraude ou não ao longo das amostras

![ver imagem](tarefa3/imagens/graf1.png)

:two:) Valores por Fraudes: indica quantidade de frandes por faixa de valor fraudado

![ver imagem](tarefa3/imagens/graf2.png)

:three:) Tempo por Fraudes: indica quantidade de frandes por faixa de tempo de ocorrência da fraude

![ver imagem](tarefa3/imagens/graf3.png)

De maneira geral as amostras são distinguíveis, pois é possível analisar as transações que são fraudulentas,
o valor dessas fraudes, o timestamp que ocorreram as fraudes, etc...

Porém, como os significado dos valores V1 a V28 não são revelados, não é possível fazer análises a respeito
das variações dos valores dessas colunas.

:mag_right: [Ver arquivo tarefa3.py](tarefa3/tarefa3.py)

## Entrega da aula 8

Vetor de Características e Distribuição do conjunto de dados

Vetor de características: [‘Time’, 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', ‘Class’]

É um problema binário: fraude ou não fraude

![ver imagem](entrega_aula_8/imagens/graf.png)

:mag_right: [Ver arquivo codigo.py](entrega_aula_8/codigo.py)
:mag_right: [Ver relatório relatorio.pdf](entrega_aula_8/relatorio.pdf)

## Tarefa 4

Cada aluno deverá apresentar o estado atual do seu projeto, mostrando e discutindo:

O dataset como um todo atributos, características, classes, amostrasdistribuição de classes
Processamento dos dados
Como foi feita a extração de características
Foi feito seleção?
Exploração e visualização de dados
Mostrar o diagrama de dispersão (scatterplot)
Mostrar visualmente o agrupamento de seu dataset com algum algoritmo de clustering
Sugestão: K-Means (mas pode usar outro, como DBScan)
Usar como número de cluster o número de classes do seu problema (se for binário, K=2)

![ver imagem](tarefa4/imagens/cluster_oficial.png)

:mag_right: [Ver arquivo clustering_v4.py](tarefa4/clustering_v4.py)
:mag_right: [Ver relatório Tarefa_4_bruno.pptx](tarefa4/Tarefa_4_bruno.pptx)

## Classificação Tarefa Final

:arrow_right: Separação do dataset em 80/20

![ver imagem](classificacao/imagens/codigoSeparacao50508020.png)

80% - 227.845 amostras
![ver imagem](classificacao/imagens/creditCard-80-print.png)

20% - 56.962 amostras
![ver imagem](classificacao/imagens/creditCard-20-print.png)



:arrow_right: Dataset muito desbalanceado: criação de 3 conjuntos reamostrados com 1 amostra fraude para cada 5 não fraude

![ver imagem](classificacao/imagens/codigoResample.png)



:arrow_right: Ajuste de limiar: descobrir os melhores parâmetros para Random Forest, KNN e SVM.

RF
![ver imagem](classificacao/imagens/codigoLimiarRF.png)


KNN
![ver imagem](classificacao/imagens/codigoLimiarKNN.png)


SVM
![ver imagem](classificacao/imagens/codigoLImiarSVM.png)



:arrow_right: Treinar modelos: rodar os algorimos com os melhores parâmetros do ajuste para 50/50 e 80/20.

RF
![ver imagem](classificacao/imagens/codigoRFconjuntos.png)

Matriz de confusão 50/50 - Conjunto 3

![ver imagem](classificacao/imagens/matriz_conf_porc_50-50-c3_RF.png)

Matriz de confusão 80/20 - Conjunto 3

![ver imagem](classificacao/imagens/matriz_conf_porc_80-20-c3_RF.png)


KNN
![ver imagem](classificacao/imagens/codigoKNNconjuntos.png)

Matriz de confusão 50/50 - Conjunto 3

![ver imagem](classificacao/imagens/matriz_conf_porc_50-50-c3_KNN.png)

Matriz de confusão 80/20 - Conjunto 3

![ver imagem](classificacao/imagens/matriz_conf_porc_80-20-c3_KNN.png)


SVM
![ver imagem](classificacao/imagens/codigoSVMconjuntos.png)

Matriz de confusão 50/50 - Conjunto 3

![ver imagem](classificacao/imagens/matriz_conf_porc_50-50-c3_SVM.png)

Matriz de confusão 80/20 - Conjunto 3

![ver imagem](classificacao/imagens/matriz_conf_porc_80-20-c3_SVM.png)




:arrow_right: KNN

70% treino e 30% testes

5 vizinhos

tempo: 273 segundos

acurácia: 0.999

recall: 0.804

![ver imagem](classificacao/matriz_conf_2_KNN.png)
![ver imagem](classificacao/matriz_conf_2_KNN_porcentagem.png)
![ver imagem](classificacao/console_KNN.png)

:mag_right: [Ver arquivo classification.py](classificacao/classification.py)


