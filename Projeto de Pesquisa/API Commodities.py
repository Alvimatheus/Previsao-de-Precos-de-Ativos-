import yfinance as yf
import pandas as pd

# Definir os símbolos dos commodities (tickers)
tickers = ['ZW=F', 'ZC=F', 'ZS=F', 'KC=F']  # Trigo, Milho, Soja, Café

# Baixar dados históricos para cada ticker e armazenar em um dicionário
dados_commodities = {}
for ticker in tickers:
    dados_commodities[ticker] = yf.download(ticker, start='2020-01-01', end='2024-11-15')

# Renomear as colunas e remover fuso horário dos índices de data
for ticker, data in dados_commodities.items():
    data.index = data.index.tz_localize(None)
    data.rename(columns={'Adj Close': ticker}, inplace=True)

# Concatenar os dados em um único DataFrame
dados_concatenados = pd.concat(dados_commodities.values(), axis=1)

# Exportar os dados para um arquivo Excel
dados_concatenados.to_excel('dados_commodities.xlsx')

# Verificar a exportação
print("Dados exportados para 'dados_commodities.xlsx' com os títulos 'Trigo', 'Milho', 'Soja' e 'Café'")