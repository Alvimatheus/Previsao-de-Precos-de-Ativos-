import wbdata
import pandas as pd
from datetime import datetime

# Definir os indicadores de interesse
indicators = {
    'SL.UEM.TOTL.ZS': 'Taxa de Desemprego (% da força de trabalho total)',
    'SL.EMP.TOTL.SP.ZS': 'Taxa de Emprego (% da população total)'
}

# Definir o intervalo de datas
data_inicio = datetime(2020, 1, 1)
data_fim = datetime(2024, 11, 15)

# Função para buscar dados e adicionar depuração
def fetch_data(indicator, data_inicio, data_fim):
    try:
        data = wbdata.get_dataframe({indicator: indicators[indicator]}, data_inicio, data_fim)
        if not data.empty:
            print(f"Dados baixados para o indicador {indicator}:")
            print(data.head())  # Adicionar depuração para verificar o conteúdo
            return data
        else:
            print(f"Erro: Dados retornados estão vazios para o indicador {indicator}")
            return None
    except Exception as e:
        print(f"Erro ao baixar os dados do indicador {indicator}: {e}")
        return None

# Baixar os dados do Banco Mundial
df_emprego = fetch_data('SL.EMP.TOTL.SP.ZS', data_inicio, data_fim)
df_desemprego = fetch_data('SL.UEM.TOTL.ZS', data_inicio, data_fim)

# Verificar se os dados foram baixados corretamente
if df_emprego is not None and df_desemprego is not None:
    # Converter os dados para DataFrames
    df_emprego = pd.DataFrame(df_emprego)
    df_desemprego = pd.DataFrame(df_desemprego)
    
    # Combinar os dados em um único DataFrame
    df = pd.concat([df_emprego, df_desemprego], axis=1)

    # Exportar os dados para um arquivo Excel
    df.to_excel('dados_emprego_desemprego.xlsx')

    # Verificar a exportação
    print("Dados exportados para 'dados_emprego_desemprego.xlsx'")
else:
    print("Não foi possível baixar todos os dados.")
