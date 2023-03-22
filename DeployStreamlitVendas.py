#!/usr/bin/env python
# coding: utf-8

# ##### Deploy do Modelo de Previsão:

# In[1]:


import pandas as pd
import joblib
import streamlit as st


# Ler a base de dados final para garantir a sequência exata das colunas/features para o modelo de previsão

# In[2]:


df = pd.read_csv('Base de Dados Final.csv')


# In[3]:


# lista da sequência de colunas, excluindo-se 'Vendas'

df = df.drop('Vendas', axis=1)
sequencia_colunas = list(df.columns)
#print(sequencia_colunas)


# Importar as features da base de dados final

# In[4]:


# print(X.columns) no código do modelo


# Criar dicionários a partir das features e seus valores iniciais

# In[5]:


# dicionário das features numéricas
x_numericos = {
    'TV':0,
    'Radio':0,
    'Jornal':0,
}


# Criar botões para cada feature (streamlit)

# In[6]:


# botões das features numéricas

for chave in x_numericos:
    valor = st.number_input(f'{chave}', step=1, value=int(0))
    x_numericos[chave] = valor
    
# botão final (previsão)  

botao = st.button('Prever Valor da Venda')

if botao:
    dataframe_X = pd.DataFrame(x_numericos, index=[0]) 
    dataframe_X = dataframe_X[sequencia_colunas] 
    modelo = joblib.load('Modelo de Previsão de Vendas.joblib') 
    preco = modelo.predict(dataframe_X) 
    st.write(f'R$ {preco[0]}') 

# sequência das linahs de código do botão:
    # criar um dataframe dos valores X com a mesma ordem de colunas que o dataframe final do projeto de ciência de dados
    # reordenar as colunas do dataframe
    # importar o modelo
    # usar a previsão do modelo
    # escrever o valor previsto no streamlit 


# Concluído o código:
# - retirar todos os displays e prints
# - exportar o arquivo para a extensão .py
# - abrir o prompt do anaconda:
#     - encontrar o local onde está o arquivo (comando cd)
#     - executar: streamlit run arquivo.py

# In[ ]:




