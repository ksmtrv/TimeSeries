import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


def readFile():
    return pd.read_csv('time.csv', delimiter=';')


def plotTimeSeries(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    plt.figure(figsize=(10, 8))
    plt.plot(df['Temperature'])

    plt.xlabel('Дата')
    plt.ylabel('Значение')
    plt.title('Временной ряд')

    plt.show()


# Построение автокорреляционной функции (ACF)
def plotACF(df):
    sm.graphics.tsa.plot_acf(df['Temperature'].dropna(), lags=9)
    plt.xlabel('Лаги')
    plt.ylabel('ACF')
    plt.title('График автокорреляции (ACF)')
    plt.show()


# Построение частной автокорреляционной функции (PACF)
def plotPACF(df):
    sm.graphics.tsa.plot_pacf(df['Temperature'].dropna(), lags=4)
    plt.xlabel('Лаги')
    plt.ylabel('PACF')
    plt.title('График частной автокорреляции (PACF)')
    plt.show()


def plotMA(df):
    # Моделирование модели скользящего среднего (MA)
    ma_model = sm.tsa.ARIMA(df['Temperature'], order=(0, 0, 1))
    ma_results = ma_model.fit()
    plt.figure(figsize=(10, 6))
    plt.plot(df['Temperature'], label='Оригинальный ряд')
    plt.plot(ma_results.predict(), label='MA(1)')
    plt.xlabel('Дата')
    plt.ylabel('Значение')
    plt.title('Модель скользящего среднего (MA)')
    plt.legend()
    plt.show()


def plotAR(df):
    #Моделирование авторегрессионной модели (AR)
    ar_model = sm.tsa.ARIMA(df['Temperature'], order=(1, 0, 0))
    ar_results = ar_model.fit()
    plt.figure(figsize=(10, 6))
    plt.plot(df['Temperature'], label='Оригинальный ряд')
    plt.plot(ar_results.predict(), label='AR(1)')
    plt.xlabel('Дата')
    plt.ylabel('Значение')
    plt.title('Авторегрессионная модель (AR)')
    plt.legend()
    plt.show()


def plotARMA(df):
    # Моделирование модели ARMA
    arma_model = sm.tsa.ARIMA(df['Temperature'], order=(1, 0, 1))
    arma_results = arma_model.fit()
    plt.figure(figsize=(10, 6))
    plt.plot(df['Temperature'], label='Оригинальный ряд')
    plt.plot(arma_results.predict(), label='ARMA(1,1)')
    plt.xlabel('Дата')
    plt.ylabel('Значение')
    plt.title('Модель ARMA')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    df = readFile()
    # plotTimeSeries(df)
    # plotACF(df)
    # plotPACF(df)
    plotAR(df)
    plotMA(df)
    plotARMA(df)

