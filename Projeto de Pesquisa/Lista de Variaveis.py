from fredapi import Fred
import pandas as pd

# Substitua 'YOUR_API_KEY' pela sua chave de API do FRED
fred = Fred(api_key='fc8bc2a9022dd09e0ce8ec07341f8364')

# Definir os códigos das séries para os indicadores desejados
# Certifique-se de usar códigos válidos
indicators = {
    'Liquidez Global do Petróleo': 'DCOILWTICO',  # Preço do Petróleo WTI
    'Taxa de Crescimento Mundial': 'NYGDPMKTPCDWLD',  # Taxa de Crescimento do PIB Mundial
    'Taxa de Juros Internacional': 'IR3TIB01USM156N'  # Taxa de Juros dos EUA (proxy para internacional)
}

# Baixar dados históricos para cada indicador e armazenar em um dicionário
dados_indicadores = {}
for nome, codigo in indicators.items():
    try:
        dados_indicadores[nome] = fred.get_series(codigo, observation_start='2020-01-01', observation_end='2024-11-15')
    except ValueError as e:
        print(f"Erro ao baixar a série {codigo}: {e}")

# Converter os dados para DataFrames e renomear as colunas
for nome, data in dados_indicadores.items():
    dados_indicadores[nome] = pd.DataFrame(data, columns=[nome])

# Concatenar os dados em um único DataFrame
dados_concatenados = pd.concat(dados_indicadores.values(), axis=1)

# Exportar os dados para um arquivo Excel
dados_concatenados.to_excel('lista_variaveis_globais_fred.xlsx')

# Verificar a exportação
print("Dados exportados para 'lista_variaveis_globais_fred.xlsx' com os títulos 'Liquidez Global do Petróleo', 'Taxa de Crescimento Mundial' e 'Taxa de Juros Internacional'")
