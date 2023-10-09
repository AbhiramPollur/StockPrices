# Stock Price Prediction Dash App

The **Stock Price Prediction Dash App** is a web application that leverages data science techniques to provide insights into historical stock prices and forecast future trends. Built using Dash, Python's web application framework, this app allows users to interactively explore stock data for different companies.

## Table of Contents

- [Installation](#installation)
- [Running the App](#running-the-app)
- [Features](#features)
- [Customization](#customization)
- [Modeling Details](#modeling-details)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Installation

Before running the app, make sure to install the required dependencies. Use the following command:

```bash
pip install pandas numpy dash plotly statsmodels scikit-learn pmdarima
```

## Running the App

To launch the Stock Price Prediction Dash App, run the following command:

```bash
python app.py
```

Access the app through your web browser by navigating to `http://127.0.0.1:8050/`.

## Features

1. **Company Selection:**
   Choose from a variety of companies through the dropdown menu. Each company has its own dataframe for a tailored analysis.

2. **Price Type Selection:**
   Customize your analysis by selecting the type of stock price to explore – Close, Open, High, or Low – using intuitive radio buttons.

3. **Historical Chart:**
   Explore the historical stock prices of the selected company and price type with an engaging line chart.

4. **Forecast Chart:**
   Utilize the Auto ARIMA model to visualize forecasted stock prices. The app evaluates the best ARIMA order for each company, ensuring accurate forecasts.

5. **Dynamic Forecast Period:**
   The app dynamically adjusts the forecast period based on the current day. If it's the weekend, the forecast extends to the next week; otherwise, it covers the next five business days.

6. **Background Image:**
   Enhance the app's visual appeal by replacing the default background image (`background.png`) with an image of your choice in the `/assets/` directory.

## Customization

- **Background Image:**
  Replace the default background image with your preferred one in the `/assets/` directory. Experiment with various images to tailor the app's visual aesthetic.

- **Styling:**
  Fine-tune the CSS styling in the app layout to align with your design preferences. Adjust font styles, color schemes, and layout dimensions to create a personalized user experience.

## Modeling Details

- **Dataframe Division:**
  The dataset undergoes meticulous division into separate dataframes for each company, allowing for a granular analysis.

- **Auto ARIMA:**
  The Auto ARIMA model is employed for forecasting stock prices across various types – High, Low, Open, Close. The model's automatic determination of the best ARIMA order ensures precision in predictions.

- **Computational Time:**
  Note that the evaluation of High, Low, Open, and Close prices for each company using Auto ARIMA is computationally intensive. The app prioritizes accuracy over speed, offering a robust forecasting mechanism.

## Dependencies

The Stock Price Prediction Dash App relies on several essential Python libraries:

- pandas
- numpy
- dash
- plotly
- statsmodels
- scikit-learn
- pmdarima

Ensure these libraries are installed in your Python environment before launching the application.

## Contributing

Contributions are welcome! If you have ideas for improvements or new features, feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
