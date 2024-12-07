from fredapi import Fred
import yfinance as yf
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import SGD
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot as plt
# Disable the warnings
import warnings
warnings.filterwarnings('ignore')

from pickle import dump
from matplotlib import pyplot



# Substitua 'YOUR_API_KEY' pela sua chave de API do FRED
fred = Fred(api_key='fc8bc2a9022dd09e0ce8ec07341f8364')

# Definir os códigos das séries para os indicadores desejados no FRED
indicators = {
    'Taxa de Juros Internacional': 'IR3TIB01USM156N'  # Taxa de Juros dos EUA (proxy para internacional)
}

# Baixar dados históricos para cada indicador do FRED e armazenar em um dicionário
dados_indicadores = {}
for nome, codigo in indicators.items():
    try:
        dados_indicadores[nome] = fred.get_series(codigo)
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
    dados_commodities[ticker] = yf.download(ticker, period='max')

# Renomear as colunas e remover fuso horário dos índices de data dos dados do Yahoo Finance
for ticker, data in dados_commodities.items():
    data.index = data.index.tz_localize(None)
    data.rename(columns={'Adj Close': ticker}, inplace=True)

# Concatenar os dados do Yahoo Finance em um único DataFrame
dados_yahoo_concatenados = pd.concat(dados_commodities.values(), axis=1)

# Criar um objeto ExcelWriter para salvar os dados em um único arquivo com duas abas
with pd.ExcelWriter('commodities_variaveis.xlsx') as writer:
    # Salvar dados do FRED na primeira aba
    dados_fred_concatenados.to_excel(writer, sheet_name='Variáveis')
    
    # Salvar dados do Yahoo Finance na segunda aba
    dados_yahoo_concatenados.to_excel(writer, sheet_name='Trigo, Milho, Soja e Café')

# Verificar a exportação
print("Dados exportados para 'commodities_variaveis.xlsx' com abas 'Variáveis' e 'Trigo, Milho, Soja e Café'")



# Carregar o arquivo Excel
file_path = r'C:\Users\user\Documents\GitHub\Previsao-de-Precos-de-Ativos-\Projeto de Pesquisa\commodities_variaveis.xlsx'  # Substitua pelo caminho correto
data = pd.read_excel(file_path, index_col=0, parse_dates=True)

# Separar as colunas alvo (Y) e preditoras (X)
Y = data['Taxa de Juros Internacional']  # Substitua 'target' pelo nome da coluna alvo
X = data.drop(columns=[])  # Remova ou ajuste de acordo com seu dataset

# Divisão treino/teste (80% treino, 20% teste)
validation_size = 0.2
train_size = int(len(X) * (1 - validation_size))

X_train, X_test = X.iloc[:train_size, :], X.iloc[train_size:, :]
Y_train, Y_test = Y.iloc[:train_size], Y.iloc[train_size:]
    
#Verificar e tratar valores inválidos
def preprocess_data(X, Y):
    # Substituir infinitos por NaN para facilitar o tratamento
    X = X.replace([np.inf, -np.inf], np.nan)
    Y = Y.replace([np.inf, -np.inf], np.nan)

    # Remover linhas com NaN
    X = X.dropna()
    Y = Y.loc[X.index]  # Garantir alinhamento após remover linhas de X
    
    return X, Y

# Aplicar pré-processamento
X_train, Y_train = preprocess_data(X_train, Y_train)
X_test, Y_test = preprocess_data(X_test, Y_test)

# Verificar se o pré-processamento removeu todos os valores inválidos
assert not X_train.isnull().any().any(), "X_train contém NaN após pré-processamento"
assert not Y_train.isnull().any(), "Y_train contém NaN após pré-processamento"


# Alinhar X_test e Y_test para garantir que os tamanhos sejam compatíveis
X_test, Y_test = X_test.align(Y_test, join='inner', axis=0)

# Verificar tamanhos e exibir informações diagnósticas
print(f"Tamanho de X_test: {X_test.shape}")
print(f"Tamanho de Y_test: {Y_test.shape}")

# Certifique-se de que os tamanhos sejam iguais após o alinhamento
assert X_test.shape[0] == Y_test.shape[0], "Os tamanhos de X_test e Y_test não são compatíveis após o alinhamento."

# Modelo ARIMA
print("Treinando ARIMA...")
modelARIMA = ARIMA(endog=Y_train, exog=X_train, order=(1, 0, 0))
model_fit = modelARIMA.fit()

# Calcular erro de treinamento
fitted_values = model_fit.fittedvalues  # Valores ajustados para o conjunto de treinamento
error_Training_ARIMA = mean_squared_error(Y_train, fitted_values)
print(f"ARIMA Training MSE: {error_Training_ARIMA}")

# Ajustar previsão com base nos tamanhos reais
start_idx = len(Y_train)
end_idx = start_idx + len(Y_test) - 1

# Previsões para o conjunto de teste
predicted_ARIMA = model_fit.predict(start=start_idx, end=end_idx, exog=X_test)

# Calcular erro de teste
error_Test_ARIMA = mean_squared_error(Y_test, predicted_ARIMA)
print(f"ARIMA Test MSE: {error_Test_ARIMA}")


# Modelo LSTM
print("Treinando LSTM...")
seq_len = 2  # Tamanho da janela

# Ajustar Y_train_LSTM e Y_test_LSTM para corresponder ao tamanho de X_train_LSTM e X_test_LSTM
Y_train_LSTM = np.array(Y_train)[seq_len - 1:]  # Remover os primeiros seq_len-1 valores
Y_test_LSTM = np.array(Y_test)[seq_len - 1:]  # Alinhar com o tamanho de X_test_LSTM

# Criar X_train_LSTM
X_train_LSTM = np.zeros((len(Y_train_LSTM), seq_len, X_train.shape[1]))
for i in range(seq_len):
    X_train_LSTM[:, i, :] = np.array(X_train)[i:len(Y_train_LSTM) + i, :]

# Criar X_test_LSTM
X_test_LSTM = np.zeros((len(Y_test_LSTM), seq_len, X_test.shape[1]))
for i in range(seq_len):
    X_test_LSTM[:, i, :] = np.array(X_test)[i:len(Y_test_LSTM) + i, :]

# Verificar tamanhos finais
print(f"Tamanho de X_train_LSTM: {X_train_LSTM.shape}")
print(f"Tamanho de Y_train_LSTM: {Y_train_LSTM.shape}")
print(f"Tamanho de X_test_LSTM: {X_test_LSTM.shape}")
print(f"Tamanho de Y_test_LSTM: {Y_test_LSTM.shape}")

# Criar e treinar o modelo LSTM
def create_LSTMmodel(neurons=12, learn_rate=0.01, momentum=0.0):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(X_train_LSTM.shape[1], X_train_LSTM.shape[2])))
    model.add(Dense(1))
    optimizer = SGD(learning_rate=learn_rate, momentum=momentum)
    model.compile(loss='mse', optimizer=optimizer)
    return model

LSTMModel = create_LSTMmodel(neurons=50, learn_rate=0.01, momentum=0.0)
LSTMModel_fit = LSTMModel.fit(
    X_train_LSTM, 
    Y_train_LSTM, 
    validation_data=(X_test_LSTM, Y_test_LSTM), 
    epochs=330, 
    batch_size=72, 
    verbose=0
)

# Plotando perdas de treinamento e validação
plt.plot(LSTMModel_fit.history['loss'], label='train')
plt.plot(LSTMModel_fit.history['val_loss'], label='test')
plt.legend()
plt.show()

# Erros de treinamento e teste para LSTM
error_Training_LSTM = mean_squared_error(Y_train_LSTM, LSTMModel.predict(X_train_LSTM))
predicted_LSTM = LSTMModel.predict(X_test_LSTM)
error_Test_LSTM = mean_squared_error(Y_test, predicted_LSTM)

# Comparando os resultados
print(f"ARIMA Training MSE: {error_Training_ARIMA}, Test MSE: {error_Test_ARIMA}")
print(f"LSTM Training MSE: {error_Training_LSTM}, Test MSE: {error_Test_LSTM}")

# Plotando valores reais vs previstos
plt.plot(np.exp(Y_test).cumprod(), 'r', label='Valores Reais')
plt.plot(np.exp(predicted_ARIMA).cumprod(), 'b', label='ARIMA - Valores Previstos')
plt.plot(np.exp(predicted_LSTM).cumprod(), 'g', label='LSTM - Valores Previstos')
plt.legend()
plt.show()


# Selecionar colunas específicas para o modelo ARIMA
# Substitua ['col1', 'col2', 'col3'] pelas colunas que você deseja usar
cols_arima = ['col1', 'col2', 'col3', 'col4']  # Adicione os nomes corretos das colunas
X_train_ARIMA = X_train[cols_arima]
X_test_ARIMA = X_test[cols_arima]

# Grid Search for ARIMA Model
def evaluate_arima_model(p_values, d_values, q_values): 
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    mse = evaluate_arima_model(order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print(f'ARIMA{order} MSE={mse:.7f}')
                except Exception as e:
                    print(f"Erro ao ajustar ARIMA{order}: {e}")
                    continue
    if best_cfg is None:
        raise ValueError("Nenhuma combinação válida de (p, d, q) foi encontrada.")
    print(f'Melhor modelo ARIMA: ARIMA{best_cfg} com MSE={best_score:.7f}')
    return best_cfg


p_values = range(0, 5)
q_values = range(0, 5)
d_values = range(0, 5)

warnings.filterwarnings("ignore")
# In[ ]
# Executar Grid Search
# Tentar executar o grid search
# Intervalos para o grid search

try:
    best_order = evaluate_arima_model(p_values, d_values, q_values)
except ValueError as e:
    print(e)
    best_order = (1, 0, 0)  # Fallback para um modelo simples

# Ajuste o modelo com o melhor conjunto de parâmetros
print(f"Treinando modelo ARIMA com a ordem {best_order}...")
modelARIMA_tuned = ARIMA(endog=Y_train, exog=X_train_ARIMA, order=best_order)
model_fit_tuned = modelARIMA_tuned.fit()
print("Modelo ARIMA ajustado com sucesso.")


best_order = evaluate_arima_model(p_values, d_values, q_values)   



# Verificar tamanhos após a seleção
print(f"Tamanho de X_train_ARIMA: {X_train_ARIMA.shape}")
print(f"Tamanho de X_test_ARIMA: {X_test_ARIMA.shape}")

# Modelo ARIMA ajustado
print("Treinando modelo ARIMA ajustado...")
modelARIMA_tuned = ARIMA(endog=Y_train, exog=X_train_ARIMA, order=best_order)
model_fit_tuned = modelARIMA_tuned.fit()

# Previsão para a amostra de teste
predicted_tuned = model_fit_tuned.predict(start=train_size, end=train_size + len(Y_test) - 1, exog=X_test_ARIMA)

# Calcular o erro na amostra de teste
error_Test_ARIMA_tuned = mean_squared_error(Y_test, predicted_tuned)
print(f"Erro MSE do modelo ARIMA ajustado no conjunto de teste: {error_Test_ARIMA_tuned}")

# Salvar o modelo ajustado
filename = 'finalized_model.sav'
dump(model_fit_tuned, open(filename, 'wb'))
print(f"Modelo ARIMA salvo como {filename}")

# Gráfico final com valores reais e previstos
plt.figure(figsize=(10, 6))
plt.plot(Y_test.index, Y_test, label="Valores Reais", color='blue')
plt.plot(Y_test.index, predicted_tuned, label="Valores Previstos (ARIMA)", color='orange')
plt.title("Comparação de Valores Reais e Previstos (Teste)")
plt.xlabel("Data")
plt.ylabel("Valores")
plt.legend(loc="upper left")
plt.show()


# In[]

tr_len = len(X_train_ARIMA)
te_len = len(X_test_ARIMA)
to_len = len(X)


modelARIMA_tuned = ARIMA(endog=Y_train, exog=X_train_ARIMA, order=[2,0,1])
model_fit_tuned = modelARIMA_tuned.fit()
predicted_tuned = model_fit_tuned.predict(start=tr_len - 1, end=to_len - 1, exog=X_test_ARIMA)[1:]
print(mean_squared_error(Y_test, predicted_tuned))

# Save the tuned ARIMA model
filename = 'finalized_model.sav'
dump(model_fit_tuned, open(filename, 'wb'))

# Ensure tz-naive indices for plotting
Y_test.index = Y_test.index.tz_localize(None)
# Atribuir o índice de Y_test ao índice de predicted_tuned
predicted_tuned.index = Y_test.index


# Plotting actual vs predicted values
pyplot.plot(np.exp(Y_test).cumprod(), 'r', label='Actual Values')
pyplot.plot(np.exp(predicted_tuned).cumprod(), 'b', label='Fitted (Predicted) Values')
pyplot.rcParams["figure.figsize"] = (8, 5)
pyplot.legend(loc='upper left')
pyplot.show()


