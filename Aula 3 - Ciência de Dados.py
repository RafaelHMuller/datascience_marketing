#!/usr/bin/env python
# coding: utf-8

# # Projeto Ciência de Dados - Módulo 40. Semana de Python
# 
# #### Projeto de Ciência de Dados:
# 
# - 1: Entendimento do Desafio
# - 2: Entendimento da Área/Empresa
# - 3: Extração/Obtenção de Dados
# - 4: Ajuste de Dados (Tratamento/Limpeza)
# - 5: Análise Exploratória (Excluir Outliers)
# - 6: Encoding + Modelagem + Algoritmos (Inteligência Artificial)
# - 7: Interpretação de Resultados
# - 8: Deploy (Visualização Final)

# ### 1: Entendimento do Desafio
# ### 2: Entendimento da Área/Empresa

# Nosso desafio é conseguir prever as vendas que vamos ter em determinado período com base nos gastos em anúncios nas 3 grandes redes que a empresa Hashtag investe: TV, Jornal e Rádio.<br>
# Base de Dados: https://drive.google.com/drive/folders/1o2lpxoi9heyQV1hIlsHXWSfDkBPtze-V?usp=sharing

# ### 3: Extração/Obtenção de Dados

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv('advertising.csv')
display(df)
df.info()
df.describe()


# ### 4: Ajuste de Dados

# In[3]:


# os valores, na realidade, estão na casa do milhar (informação dada pela empresa)
for coluna, valor in df.items():
    if coluna == 'Vendas':
        valor *= 100000
    else:
        valor *= 1000

display(df)


# ### 5: Análise Exploratória

# Como estão distribuidos os valores investidos entre as formas de marketing ?

# In[4]:


plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
plt.title("Distribuição dos valores em TV")
sns.boxplot(data=df, x="TV")
plt.subplot(1,3,2)
plt.title("Distribuição dos valores em Radio")
sns.boxplot(data=df, x="Radio")
plt.subplot(1,3,3)
plt.title("Distribuição dos valores em Jornal")
sns.boxplot(data=df, x="Jornal")

plt.figure(figsize=(15,5))
plt.title("Distribuição dos valores em Vendas")
sns.boxplot(data=df, x="Vendas")


# Qual a correlação entre as formas de marketing e as vendas?

# In[5]:


plt.figure(figsize=(8,5))
plt.title('Correlação entre as formas de marketing e as vendas')
sns.heatmap(df.corr(), cmap ='RdBu_r', annot =True)
plt.show()


# Storytelling dos dados

# In[6]:


total_TV = df['TV'].sum()
total_Radio = df['Radio'].sum()
total_Jornal = df['Jornal'].sum()
total_Vendas = df['Vendas'].sum()

total_marketing = total_TV+total_Radio+total_Jornal

porcentagem_TV = (total_TV/total_marketing)
porcentagem_Radio = (total_Radio/total_marketing)
porcentagem_Jornal = (total_Jornal/total_marketing)

lucro = total_Vendas - total_marketing

print(f'''Ao longo do período, aplicou-se R$ {total_marketing:,.0f} em marketing:
{porcentagem_TV:.0%} em TV, {porcentagem_Radio:.0%} em Rádio e {porcentagem_Jornal:.0%} em Jornal.

O lucro bruto com as vendas no período foi de R$ {total_Vendas:,.0f}, o que representa um lucro líquido de {lucro:,.0f}. 
''')


# ### 6: Encoding + Modelagem + Algoritmos

# ###### Encoding
#     - Ajustar todos os dados/features para facilitar o trabalho do modelo; para isso, deve-se codificar os dados de texto em números;
#     - Features de Valores True ou False - substitui-se True por 1 e False por 0
#     - Features de Categoria/Textos - utiliza-se o método de encoding de variáveis dummies (pegam-se todas as categorias da coluna, transforma-se cada categoria em coluna e atribuem-se os valores 1 ou 0)

# In[7]:


# Todos os dados já são numéricos, logo não há necessidade de encoding


# In[8]:


df_copia = df.copy()


# ######  Modelagem + Algoritmos

# 7 etapas para aplicar a Inteligência Artificial/Machine Learning:
# 1. definir se é classificação ou regressão
# 2. escolher as métricas de avaliação
# 3. escolher os modelos a serem testados
# 4. treinamento dos modelos (machine learning)
# 5. comparar os resultados dos modelos e escolher o melhor
# 6. analisar o melhor modelo mais a fundo
# 7. fazer os ajustes do melhor modelo

# ###### 1. classificação ou regressão
# - Classificação: objetivo é classificar/criar categorias (ex: categorias A, B, C);
# - Regressão: objetivo é encontrar um valor específico (ex: preço).

# In[9]:


# Nosso modelo de previsão deve prever o valor de vendas a partir dos investimentos em amrketing, logo deve ser um modelo de previsão.


# ##### 2. métricas de avaliação
# - Métrica R²: o valor resultante mede o quanto o modelo consegue explicar a variância dos dados, a capacidade de previsão correta;
# - Métrica RSME (Erro Quadrático Médio): o valor resultante mede o quanto o modelo erra.

# In[10]:


def avaliacao_modelo(modelo, y_test, previsao):
    '''Parâmetros:
    modelo: nome do modelo escolhido,
    y_teste: valores a serem testados (no caso, o preço real dos imóveis),
    previsao: valor previsto
    '''
    r2 = r2_score(y_test, previsao)
    RSME = np.sqrt(mean_squared_error(y_test, previsao))
    return f'Modelo {modelo}:\nR² = {r2:.2%}\nRSME = {RSME:.2f}\n\n'


# ##### 3. escolher os modelos 
#         - Regressão Linear: este modelo traça a melhor reta que minimiza os erros dos dados/valores; é um modelo muito rápido, porém não existe uma reta que minimize todos os erros, deixando alguns dados muito fora do valor médio;
#         - Árvore de Decisão: este modelo de árvores de decisão pode ser correlacionado com o jogo do Akinator; este modelo cria várias árvores de decisão (forest) com partes aleatórias menores da base de dados e, a cada árvore, os dados vão sendo filtrados, por meio da melhor pergunta/filtro possível logo no início, até o melhor valor possível; no fim, cria-se uma média destes valores;
#         - Extra Trees: este modelo de árvores de decisão é parecido com o anterior, porém a diferença principal é que as perguntas/os filtros feitas pelo modelo para alcançar o melhor valor final são aleatórias.

# In[11]:


modelo_lr = LinearRegression()
modelo_rf = RandomForestRegressor()
modelo_et = ExtraTreesRegressor()

dicionario_modelos = {
    'LinearRegression': modelo_lr,
    'RandomForest': modelo_rf,
    'ExtraTrees': modelo_et,
}


# ##### 4. treinamento dos modelos / machine learning
#         Separa-se a base de dados em Treino e Teste:
#         - Treino: parte dos dados usados para o modelo aprender (proporção maior, 80-90%);
#         - Teste: parte dos dados usados para o modelo prever (proporção menor, 20-10%); verifica-se se o modelo funciona bem em uma situação real.
#         
#         Train_test_split: por padrão, essa função do sklearn seleciona aleatoriamente dados de X e Y e os separa em dados de treino e teste; ao aplicar o parâmetro random_state, essa seleção é sempre a mesma, ou seja, os dados de treino e teste serão sempre os mesmos.
#         
#         Os modelos usam os dados de treino (X_train, y_train) para treinar (.fit) e, então, usam os dados de teste (X_test) para tentar prever (.predict) o Y_test.

# In[12]:


# criam-se as variáveis X (dados de todas as colunas/features, menos a que será prevista) e Y (valores a serem previstos)

y = df_copia['Vendas']
X = df_copia.drop('Vendas', axis=1)

# separar os dados em treino e teste

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3, shuffle=True)

# treinar e testar os modelos

for nome_modelo, modelo in dicionario_modelos.items():
    #treino
    inicio = time.time()
    modelo.fit(X_train, y_train)
    
    #teste
    previsao = modelo.predict(X_test)
    fim = time.time()
    
    print(f'{avaliacao_modelo(nome_modelo, y_test, previsao)}Timer: {fim-inicio:.2f} seconds\n\n')
    


# ##### 5. escolher o melhor modelo
#         A partir das métricas de avaliação escolhidas (2), comparam-se os modelos (3) e escolhe-se o melhor: o modelo com o maior valor de R² + menor valor de RSME + menor tempo de previsão + menor número de dados necessários + outros.

# In[13]:


# pelas métricas de avaliação R² e RSME e pela velocidade de treino e previsão, o melhor modelo é:

#Modelo ExtraTrees:
#R² = 93.40%
#RSME = 140930.40
#Timer: 0.08 seconds


# ##### 6. analisar o melhor modelo
#         - Verifica-se gráfico scatterplot dos valores previstos pelo modelo;
#         - Identifica-se a importância de cada feature (.feature_importances_) para tentar aprimorar o modelo; modelos que usam menos informações são, teoricamente, mais eficientes na previsão.

# In[14]:


# visualização gráfica scatterplot e lineplot do modelo vencedor

#scatterplot
modelo_et.fit(X_train, y_train)
previsao = modelo_et.predict(X_test)

fig, ax = plt.subplots(figsize=(5,5))
ax.set_title("Valor em Vendas")
ax.set_ylabel('Valor em Vendas no Teste',fontsize=12)
ax.set_xlabel('Valor em Vendas no Teste',fontsize=12)
ax.scatter(y_test,previsao)

#lineplot
df_resultado = pd.DataFrame()
df_resultado['y_teste'] = y_test
df_resultado['y_previsao'] = previsao
df_resultado = df_resultado.reset_index(drop=True)

fig, ax2 = plt.subplots(figsize=(15,5))
ax2.set_title("Valor em Vendas")
ax2.set_ylabel('Valor em Vendas no Teste',fontsize=12)
ax2.set_xlabel('Valor em Vendas no Teste',fontsize=12)
sns.lineplot(data=df_resultado)


# In[15]:


# o método .feature_importances_ do modelo escolhido traz a importância de cada feature na mesma ordem dos dados de treino/teste

lista_valores = modelo_et.feature_importances_
lista_colunas = X_train.columns

df_importancia_features = pd.DataFrame(index=lista_colunas, data=lista_valores)
df_importancia_features = df_importancia_features.sort_values(by=0, ascending=False)
display(df_importancia_features)

plt.figure(figsize=(5,5))
plt.title('Importância de cada feature para o modelo de previsão')
sns.barplot(x=df_importancia_features.index, y=df_importancia_features[0])


# ###### 7. ajustar o melhor modelo
#         A partir da análise anterior (6), altera-se a base de dados (ex: excluir coluna, retirar outlier, recuperar outliers excluídos, etc.) e, a cada etapa, treina-se e testa-se o modelo novamente, para comparar a avaliação do modelo (R² e RSME) antes e depois da mudança, sempre lembrando que modelos que usam menos informações são, teoricamente, mais eficientes na previsão.
#         Dica: antes de alterar a base de dados original ao realizar os testes de ajuste do modelo, melhor criar uma cópia; caso o ajuste não for bom, teria que rodar todo o código do início para atualizar a base de dados original.

# In[16]:


# como são apenas 3 features, não vou excluir nenhum dado, mantendo-se o modelo inalterado


# ### 8: Deploy
#     1. exportar a base de dados atualizada 
#     2. criar arquivo do modelo (joblib)
#     3. escolher a forma de deploy:
#         - arquivo executável (tkinter)
#         - criar um microsite na internet (flask)
#         - criar um microsite para uso direto (streamlit)
#     4. abrir outro arquivo python para criar o código do deploy
#     5. atribuir ao botão o carregamento do modelo

# In[17]:


#1

df_copia.to_csv('Base de Dados Final.csv', index=False)


# In[18]:


#2

import joblib
joblib.dump(modelo_et, 'Modelo de Previsão.joblib')


# In[19]:


#3,4,5

# o arquivo do jupyter deve estar na mesma pasta onde está o modelo (.joblib)
# o passo a passo da criação do deploy pela biblioteca streamlit está no outro arquivo
# após a criação de todo o deploy pelo streamlit, o arquivo deverá ser exportado como .py para rodar


# In[ ]:




