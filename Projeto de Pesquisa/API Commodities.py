from fredapi import Fred
import yfinance as yf
import pandas as pd

# Substitua 'YOUR_API_KEY' pela sua chave de API do FRED; FEDERAL RESERVE ECONOMIC DATA 
fred = Fred(api_key='fc8bc2a9022dd09e0ce8ec07341f8364')

# Definir os códigos das séries para os indicadores desejados no FRED
indicators = {
    'Liquidez Global do Petróleo': 'DCOILWTICO',  # Preço do Petróleo WTI
    'Taxa de Crescimento Mundial': 'NYGDPMKTPCDWLD',  # Taxa de Crescimento do PIB Mundial
    'Taxa de Juros Internacional': 'IR3TIB01USM156N'  # Taxa de Juros dos EUA (proxy para internacional)
}

# Baixar dados históricos para cada indicador do FRED e armazenar em um dicionário
dados_indicadores = {}
for nome, codigo in indicators.items():
    try:
        dados_indicadores[nome] = fred.get_series(codigo, observation_start='2020-01-01', observation_end='2024-11-15')
    except ValueError as e:
        print(f"Erro ao baixar a série {codigo}: {e}")

# Converter os dados do FRED para DataFrames e renomear as colunas
for nome, data in dados_indicadores.items():
    dados_indicadores[nome] = pd.DataFrame(data, columns=[nome])

# Concatenar os dados do FRED em um único DataFrame
dados_fred_concatenados = pd.concat(dados_indicadores.values(), axis=1)

# Definir os símbolos dos commodities (tickers) para o Yahoo Finance
tickers = ['ZW=F', 'ZC=F', 'ZS=F', 'KC=F']  # Trigo, Milho, Soja, Café

# Baixar dados históricos para cada ticker do Yahoo Finance e armazenar em um dicionário
dados_commodities = {}
for ticker in tickers:
    dados_commodities[ticker] = yf.download(ticker, start='2020-01-01', end='2024-11-15')

# Renomear as colunas e remover fuso horário dos índices de data dos dados do Yahoo Finance
for ticker, data in dados_commodities.items():
    data.index = data.index.tz_localize(None)
    data.rename(columns={'Adj Close': ticker}, inplace=True)

# Concatenar os dados do Yahoo Finance em um único DataFrame
dados_yahoo_concatenados = pd.concat(dados_commodities.values(), axis=1)

# Criar um objeto ExcelWriter para salvar os dados em um único arquivo com duas abas
with pd.ExcelWriter('dados_combinados.xlsx') as writer:
    # Salvar dados do FRED na primeira aba
    dados_fred_concatenados.to_excel(writer, sheet_name='Variáveis')
    
    # Salvar dados do Yahoo Finance na segunda aba
    dados_yahoo_concatenados.to_excel(writer, sheet_name='Trigo, Milho, Soja e Café')

# Verificar a exportação
print("Dados exportados para 'dados_combinados.xlsx' com abas 'Variáveis' e 'Trigo, Milho, Soja e Café'")
