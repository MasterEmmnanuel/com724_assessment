from flask import Flask, render_template, request, jsonify
import pandas_datareader.data as web
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
import joblib

app = Flask(__name__)

coins = ['BTC', 'ETH', 'LTC', 'XRP', 'BCH', 'ADA', 'XEM', 'TRX', 'XLM', 'MIOTA', 'DASH', 'EOS', 'XMR',
         'NEO', 'BTG', 'ETC', 'LSK', 'ICX', 'ZEC', 'XVG', 'OMG', 'DOGE', 'BNB', 'STEEM']


def get_cryptocurrency(coin_name):
    df = pd.DataFrame()
    crypt = web.DataReader(f"{coin_name}-USD", 'yahoo', '2021-05-01', '2022-07-20')
    coin_rank = coins.index(coin_name)
    print(coin_rank)
    crypt['rank'] = coin_rank
    df = df.append(crypt)
    return df


def data_graph(target, data):
    time = str(data.tail(1).index.values[0]).split("T")[0]
    df = data[-target:]['target'].values

    df = pd.DataFrame({'Close': df}, index=pd.date_range(time, periods=target, freq='D'))

    get_plot = pd.concat([data[['Close']], df], sort=False)

    plt.figure(figsize=(12, 6))
    plt.grid(True)
    plt.xlabel('Dates')
    plt.ylabel('Crypto Price')
    plt.plot(get_plot[:-target][['Close']], 'green', label='Price (USD)')
    plt.plot(get_plot[-(target+1):][['Close']], 'orange', label='Target')
    plt.legend()
    # plt.show()
    plt.gcf().autofmt_xdate()
    plt.savefig('static/completedGraph.jpg', format='jpg')


@app.route('/flask_prediction', methods=['POST'])
def myPrediction():
    coin_name = request.form['coin_name']
    target = int(request.form['target'])

    data_df = get_cryptocurrency(coin_name)
    data_df.drop('Adj Close', axis=1, inplace=True)
    copy_data_df = data_df.copy()
    copy_data_df['target'] = copy_data_df['Close']

    # build scale data using csv file
    df = pd.read_csv('data_file.csv')
    df['Date'] = pd.to_datetime(df.Date)
    df['target'] = df['Close']
    df.index = df.Date
    df.drop(['Date', 'Adj Close', 'coin'], axis=1, inplace=True)
    # data scaling
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaled_df = scaler.fit_transform(df)
    scaled_data_df = scaler.transform(copy_data_df)
    df_columns = ['High', 'Low', 'Open', 'Close', 'Volume', 'rank', 'target']
    scaled_data_df = pd.DataFrame(scaled_data_df, columns=df_columns)
    #     print(scaled_data_df.head())
    # predict
    rmf = joblib.load('model.sav')
    input_data = scaled_data_df.drop(['Close', 'target'], axis=1)
    scaled_data_df['target'] = rmf.predict(input_data)

    scaled_data_df = scaler.inverse_transform(scaled_data_df)
    scaled_data_df = pd.DataFrame(scaled_data_df, columns=df_columns)
    scaled_data_df.index = data_df.index
    data_df['target'] = scaled_data_df['target']

    price = data_df['Close'].tail(1).values[0]
    predicted = data_df['target'].tail(1).values[0]

    # plot graph
    data_graph(target, data_df)

    return jsonify(price=price, predicted=predicted)


@app.route('/')
def hello_world():
    return render_template('html_file.html', crypto=coins)


app.run(host='127.0.0.1', port=5000)
