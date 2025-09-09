# Bitcoin-Price-Prediction
Bitcoin Price Prediction Using Machine Learning in Python

## Overview

This project aims to predict Bitcoin prices using machine learning models. By leveraging historical Bitcoin price data and various features, the model attempts to forecast future prices accurately. The project involves data preprocessing, exploratory data analysis (EDA), feature engineering, model selection, and evaluation.

## Dataset

The dataset consists of historical Bitcoin price data obtained from a reliable source, including features such as:

* **Date**: Timestamp of the recorded price.
* **Open Price**: Price at the opening of the trading session.
* **High Price**: Highest price recorded during the session.
* **Low Price**: Lowest price recorded during the session.
* **Close Price**: Price at the close of the trading session.
* **Volume**: Amount of Bitcoin traded during the session.
* **Market Cap**: Total market capitalization of Bitcoin.

## Technologies Used

* **Python** (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
* **Machine Learning Models** (Linear Regression, Random Forest, LSTM, ARIMA)
* **Data Visualization** (Matplotlib, Seaborn)
* **Jupyter Notebook**

## Project Workflow

1. **Data Collection & Preprocessing**

   * Load the dataset and handle missing values.
   * Convert date columns to datetime format.
   * Normalize/standardize numerical features.
2. **Exploratory Data Analysis (EDA)**

   * Analyze trends, seasonality, and volatility.
   * Visualize price trends and correlation between variables.
3. **Feature Engineering**

   * Create new features like moving averages, RSI, and Bollinger Bands.
   * Encode categorical variables (if any).
4. **Model Selection & Training**

   * Train multiple models (Linear Regression, Random Forest, LSTM, ARIMA).
   * Tune hyperparameters using GridSearchCV or other techniques.
5. **Model Evaluation**

   * Assess models using RMSE, MAE, and R² scores.
   * Compare performance and select the best model.
6. **Prediction & Deployment**

   * Make future price predictions.
   * (Optional) Deploy model using Flask or Streamlit.

## Results

* The model with the best performance (based on evaluation metrics) is selected.
* Predictions are compared with actual values using visual plots.

## Conclusion

This project demonstrates the use of machine learning to predict Bitcoin prices based on historical data. While models like LSTM can capture long-term trends effectively, traditional models like Random Forest provide robust predictions. Future improvements can involve incorporating external factors such as news sentiment and macroeconomic indicators.

## Future Enhancements

* Incorporate sentiment analysis of crypto news and social media.
* Use advanced deep learning models like Transformers.
* Implement real-time prediction and dashboard visualization.

## How to Run

1. Clone this repository using `git clone https://github.com/hridaymehta10/bitcoin-price-prediction.git`
2. Install dependencies using `pip install -r requirements.txt`.
3. Run `jupyter notebook` and open the project file.
4. Train the model and make predictions.


## License

This project is open-source and available under the MIT License.
