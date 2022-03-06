# CI1030_Bruno_Farias_20212
Repositório criado para as tarefas da disciplina Ciência de Dados para Segurança - UFPR - 2021-2
Bruno E. Farias - GRR20186715

<p align="center">
<img src="http://img.shields.io/static/v1?label=STATUS&message=EM%20DESENVOLVIMENTO&color=GREEN&style=for-the-badge"/>
</p>

### Tarefas
:point_right: [Tarefa 1](#tarefa-1)

:new: [Tarefa 2](#tarefa-2)

:new: [Tarefa 3](#tarefa-3)


## Tarefa 1 :new:
Contador de pacotes/sessões de um arquivo .pcap.

:mag_right: [Ver arquivo conta_sessoes.py](tarefa1/conta_sessoes.py)

## Tarefa 2
.txt entregue na tarefa 2

:mag_right: [Ver arquivo Bruno_Farias.txt](tarefa2/Bruno_Farias.txt)

## Tarefa 3
:credit_card: Repositório: Fraudes em Cartões de Crédito https://www.kaggle.com/samkirkiles/credit-card-fraud/data

Dados:

:receipt: Quantidades de Amostras: 284.807

Classes:

:clock1230: Time: quantidade de tempo em segundos que passou desde a primeira transação.

:question: V1-V28: componentes obtidos com Análise de Componentes Principais (PCA - Principal Component Analysis) a respeito das transações. (O significado desses campos não é revelado por motivos de segurança).

:euro: Amount é a quantia de dinheiro envolvida na transação.

:warning: Class indica se houve fraude nessa transação (1 se houve, 0 caso contrário).

Contagem:

:arrow_right: Amostras COM Fraude: 492

:arrow_right: Amostras SEM Fraude: 284315

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
