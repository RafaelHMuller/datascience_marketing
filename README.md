<h1 align="center">
📄<br>README - Projeto Ciência de Dados Marketing
</h1>

## Índice 

* [Descrição do Projeto](#descrição-do-projeto)
* [Funcionalidades e Demonstração da Aplicação](#funcionalidades-e-demonstração-da-aplicação)
* [Pré requisitos](#pré-requisitos)
* [Execução](#execução)
* [Bibliotecas](#bibliotecas)

# Descrição do projeto
> Este repositório é meu primeiro projeto Python de ciência de dados. O objetivo deste projeto foi, a partir de uma base de dados da empresa fictícia Hashtag, realizar análise de dados dos gastos desta empresa com marketing e, por meio da ciência de dados, prever as vendas desta empresa em determinado período com base em seus gastos com marketing. O melhor modelo de previsão testado obteve mais de 92% de capacidade de acerto.

# Funcionalidades e Demonstração da Aplicação
Previsão das vendas da empresa Hashtag com base nos gastos com anúncios nas grandes redes de marketing que esta empresa investe: TV, Jornal e Rádio.

Métricas de avaliação dos modelos de previsão testados:<br>
![Screenshot_2](https://user-images.githubusercontent.com/128300382/227045972-4ba37190-e080-4854-b998-98c532eb2546.png)

Deploy do projeto (via streamlit):<br>
![Screenshot_1](https://user-images.githubusercontent.com/128300382/227044904-5925b736-19ac-4ffd-9898-2e63421004d4.png)


## Pré requisitos

* Sistema operacional Windows
* IDE de python (ambiente de desenvolvimento integrado de python)
* Base de dados (planilha csv)
* Prompt de comando do programa python (onde será executado o comando do streamlit para visualizar o deploy do projeto e, consequentemente, a previsão do valor de venda) 

## Execução

* Primeiramente deve-se executar o arquivo 'Aula 3 - Ciência de Dados.py', no qual faz-se toda a ciência de dados, machine learning e definição do melhor modelo de previsão;
*  Em seguida, executa-se o arquivo 'DeployStreamlitVendas.py', onde será gerado o deploy do modelo de previsão;
*  Por fim, dentro do prompt de comando do programa python escolhido (no meu caso, utilizei o Anaconda Prompt (anaconda3)), encontrar a pasta onde se encontra o arquivo "DeployStreamlitVendas.py" e executar o comando do streamlit run "DeployStreamlitVendas.py" para visualizar, no navegador padrão do computador, o modelo preditivo do valor de venda da empresa Hashtag.

## Bibliotecas
* pandas: biblioteca que permite, no caso, a integração de arquivo excel
* seaborn, matplotlib.pyplot, plotly.express: bibliotecas de visualização gráfica
* time: biblioteca de gerenciamento de tempo no código
* sklearn: biblioteca de machine learning/inteligência artificial
* joblib: biblioteca de criação do deploy do modelo de previsão
* streamlit: biblioteca de visualização do deploy do modelo de previsão no navegador padrão
