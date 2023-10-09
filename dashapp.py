import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pmdarima import auto_arima
import statsmodels.api as sm
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import datetime


nifty_data = pd.read_csv(r"C:\Users\Abhiram P\Desktop\WissenRC\NIFTY50_all.csv")
df2 = nifty_data
df2['Trades'] = df2['Trades'].fillna(nifty_data['Trades'].mean())
df2['Deliverable Volume'] = nifty_data['Trades'].fillna(nifty_data['Deliverable Volume'].mean())
df2['%Deliverble'] = nifty_data['%Deliverble'].fillna(nifty_data['%Deliverble'].mean())

unique_companies = df2['Symbol'].unique()

company_datasets = {}

for company in unique_companies:
    company_df = df2[df2['Symbol'] == company]
    company_datasets[company] = company_df


companies = {
    'MUNDRAPORT': company_datasets['MUNDRAPORT'],
    'ADANIPORTS': company_datasets['ADANIPORTS'],
    'ASIANPAINT': company_datasets['ASIANPAINT'],
    'UTIBANK': company_datasets['UTIBANK'],
    'AXISBANK': company_datasets['AXISBANK'],
    'BAJAJ-AUTO': company_datasets['BAJAJ-AUTO'],
    'BAJAJFINSV': company_datasets['BAJAJFINSV'],
    'BAJAUTOFIN': company_datasets['BAJAUTOFIN'],
    'BAJFINANCE': company_datasets['BAJFINANCE'],
    'BHARTI': company_datasets['BHARTI'],
    'BHARTIARTL': company_datasets['BHARTIARTL'],
    'BPCL': company_datasets['BPCL'],
    'BRITANNIA': company_datasets['BRITANNIA'],
    'CIPLA': company_datasets['CIPLA'],
    'COALINDIA': company_datasets['COALINDIA'],
    'DRREDDY': company_datasets['DRREDDY'],
    'EICHERMOT': company_datasets['EICHERMOT'],
    'GAIL': company_datasets['GAIL'],
    'GRASIM': company_datasets['GRASIM'],
    'HCLTECH': company_datasets['HCLTECH'],
    'HDFC': company_datasets['HDFC'],
    'HDFCBANK': company_datasets['HDFCBANK'],
    'HEROHONDA': company_datasets['HEROHONDA'],
    'HEROMOTOCO': company_datasets['HEROMOTOCO'],
    'HINDALCO': company_datasets['HINDALCO'],
    'HINDLEVER': company_datasets['HINDLEVER'],
    'HINDUNILVR': company_datasets['HINDUNILVR'],
    'ICICIBANK': company_datasets['ICICIBANK'],
    'INDUSINDBK': company_datasets['INDUSINDBK'],
    'INFOSYSTCH': company_datasets['INFOSYSTCH'],
    'INFY': company_datasets['INFY'],
    'IOC': company_datasets['IOC'],
    'ITC': company_datasets['ITC'],
    'JSWSTL': company_datasets['JSWSTL'],
    'KOTAKMAH': company_datasets['KOTAKMAH'],
    'KOTAKBANK': company_datasets['KOTAKBANK'],
    'LT': company_datasets['LT'],
    'M&M': company_datasets['M&M'],
    'MARUTI': company_datasets['MARUTI'],
    'NESTLEIND': company_datasets['NESTLEIND'],
    'NTPC': company_datasets['NTPC'],
    'ONGC': company_datasets['ONGC'],
    'POWERGRID': company_datasets['POWERGRID'],
    'RELIANCE': company_datasets['RELIANCE'],
    'SBIN': company_datasets['SBIN'],
    'SHREECEM': company_datasets['SHREECEM'],
    'SUNPHARMA': company_datasets['SUNPHARMA'],
    'TELCO': company_datasets['TELCO'],
    'TATAMOTORS': company_datasets['TATAMOTORS'],
    'TISCO': company_datasets['TISCO'],
    'TATASTEEL': company_datasets['TATASTEEL'],
    'TCS': company_datasets['TCS'],
    'TECHM': company_datasets['TECHM'],
    'TITAN': company_datasets['TITAN'],
    'ULTRACEMCO': company_datasets['ULTRACEMCO'],
    'UNIPHOS': company_datasets['UNIPHOS'],
    'UPL': company_datasets['UPL'],
    'SESAGOA': company_datasets['SESAGOA'],
    'SSLT': company_datasets['SSLT'],
    'VEDL': company_datasets['VEDL'],
    'WIPRO': company_datasets['WIPRO'],
    'ZEETELE': company_datasets['ZEETELE'],
    'ZEEL': company_datasets['ZEEL'],
}

# Hyperparameters
forecast_period = 30  # Adjust as needed
scaler = MinMaxScaler()

def forecast_and_evaluate(data, forecast_period):
    forecast_results = {}

    for price_type in ['Open', 'High', 'Low', 'Close']:
        y_prices = data[price_type].values
        y_prices_scaled = scaler.fit_transform(y_prices.reshape(-1, 1))

        price_model = auto_arima(
            y_prices_scaled,
            seasonal=True,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            max_order=None,
            trace=True
        )

        price_best_order = price_model.get_params()['order']

        price_final_model = sm.tsa.ARIMA(
            y_prices_scaled,
            order=price_best_order
        )
        price_final_model = price_final_model.fit()
        price_forecast = price_final_model.forecast(steps=forecast_period)
        price_forecast = price_forecast.reshape(-1, 1)
        price_forecast = scaler.inverse_transform(price_forecast)

        actual_prices = data.tail(forecast_period)[price_type].values

        price_mae = mean_absolute_error(actual_prices, price_forecast)
        price_mse = mean_squared_error(actual_prices, price_forecast)
        price_rmse = np.sqrt(price_mse)

        def mean_absolute_percentage_error(y_true, y_pred):
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        price_mape = mean_absolute_percentage_error(actual_prices, price_forecast)

        forecast_results[price_type] = {
            'forecast': price_forecast,
            'mae': price_mae,
            'mse': price_mse,
            'rmse': price_rmse,
            'mape': price_mape
        }

    return forecast_results



# Dash App
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

# Layout of the app
app.layout = html.Div(style={'font-family': 'Arial, sans-serif', 'max-width': '100%', 'margin': 'auto', 'background-color': '#f4f4f4', 'margin-left': '0', 'margin-right': '0'}, children=[
    html.H1("Stock Price Prediction", style={'text-align': 'center', 'margin-bottom': '20px'}),

    dcc.Dropdown(
        id='company-dropdown',
        options=[{'label': company, 'value': company} for company in company_datasets.keys()],
        value='MUNDRAPORT',  # Default company
        multi=False,
        style={'width': '100%', 'margin-bottom': '20px'}
    ),

    # Radio items to select price type
    dcc.RadioItems(
        id='price-type-radio',
        options=[
            {'label': 'Close', 'value': 'Close'},
            {'label': 'Open', 'value': 'Open'},
            {'label': 'High', 'value': 'High'},
            {'label': 'Low', 'value': 'Low'}
        ],
        value='Close',  # Default to 'Close'
        labelStyle={'display': 'block', 'margin-bottom': '10px'}
    ),

    dcc.Loading(
        id="Analyzing and Loading Forecast",
        type="circle",
        children=[
            dcc.Graph(id='historical-chart'),
            dcc.Graph(id='forecast-chart')
        ],
        className="loading-circle-text"
    )

])

# Callbacks to update historical and forecast charts based on selected company and price type
@app.callback(
    [Output('historical-chart', 'figure'),
     Output('forecast-chart', 'figure')],
    [Input('company-dropdown', 'value'),
     Input('price-type-radio', 'value')],
    prevent_initial_call=True
)
def update_charts(selected_company, selected_price_type):
    historical_data = company_datasets[selected_company]
    historical_prices = historical_data[selected_price_type]

    # Check if the current day is a weekend
    current_date = datetime.datetime.now()
    is_weekend = current_date.weekday() in [5, 6]  # 5 is Saturday, 6 is Sunday

    # Set the forecast period based on whether it's a weekend or not
    forecast_period = 7 if is_weekend else 5

    # Create historical chart
    historical_figure = go.Figure()
    historical_figure.add_trace(go.Scatter(x=historical_data.index, y=historical_prices, mode='lines',
                                           name=f'{selected_company} - {selected_price_type} Prices'))
    historical_figure.update_layout(title=f'Historical {selected_price_type} Prices for {selected_company}',
                                    xaxis_title='Date',
                                    yaxis_title='Price (INR)',
                                    showlegend=True,
                                    template='plotly_dark')  # Using a dark template for better visibility

    # Get forecast results
    forecast_results = forecast_and_evaluate(historical_data, forecast_period)

    # Create forecast chart
    forecast_figure = go.Figure()

    for price_type in ['Open', 'High', 'Low', 'Close']:
        forecast_data = forecast_results[price_type]['forecast'].flatten()
        forecast_figure.add_trace(go.Scatter(x=pd.date_range(end=historical_data.index[-1], periods=forecast_period, freq='D'), y=forecast_data, mode='lines',
                                             name=f'{selected_company} - {price_type} Forecast'))

    forecast_figure.update_layout(title=f'Forecast for {selected_company}',
                                  xaxis_title='Date',
                                  yaxis_title='Price (INR)',
                                  showlegend=True,
                                  template='plotly_dark')

    return historical_figure, forecast_figure


if __name__ == '__main__':
    app.run_server(debug=True)
