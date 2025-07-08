import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from statsmodels.tsa.seasonal import STL

# from airflow import DAG
# from airflow.operators.python_operator import PythonOperator
# from airflow.operators.trigger_dagrun import TriggerDagRunOperator
# from src_airflow.dags.predict_ML.path import PredictMLPathDataML, PredictMLPathDataSEASON


# Функция для загрузки данных
def load_data(season_path='predict/WB_season.csv', sales_path='predict/WB_data_future.csv'):
    pml_season = Path(__file__).resolve().parent / 'data' / 'SEASON'
    # pml_ml = PredictMLPathDataML()
    df_s = pd.read_csv(season_path)
    df = pd.read_csv(sales_path)

    # Сортировка данных
    df_s = df_s.sort_values(by=['period'])
    df = df.sort_values(by=['date'])

    # Преобразование столбцов с датами в формат datetime
    df_s['period'] = pd.to_datetime(df_s['period'])
    df['date'] = pd.to_datetime(df['date'])

    return df_s, df


# Функция для подготовки данных
def preprocess_data(df_s, df):
    # Выбираем нужные колонки
    columns_s = ['period', 'sales', 'category']
    df_s = df_s[columns_s]

    # Добавляем столбец с днём недели
    df_s['day_of_week'] = df_s['period'].dt.dayofweek
    df['day_of_week'] = df['date'].dt.dayofweek

    # Получаем список категорий, присутствующих в обоих DataFrame
    categories_hist = set(df_s['category'].unique())
    categories_pred = set(df['category'].unique())
    common_categories = categories_hist.intersection(categories_pred)

    return common_categories, df_s, df


# Функция для выполнения STL-декомпозиции
def multiplicative_stl(series, period):
    log_series = np.log(series.replace(0, np.nan).dropna())
    stl = STL(log_series, period=period, robust=True)
    res = stl.fit()
    trend = np.exp(res.trend)
    seasonal = np.exp(res.seasonal)
    resid = np.exp(res.resid)
    return trend, seasonal, resid


# Функция для обработки сезонных факторов по категории с индивидуальной корректировкой каждого артикула
def process_category(category, df_s, df, seasonal_factors_historical, period=7):
    print(f'Обработка категории: {category}')

    # Расчет агрегированной сезонности по историческим данным для категории
    df_hist_cat = df_s[df_s['category'] == category].copy()
    df_hist_cat = df_hist_cat.groupby('period').agg({'sales': 'sum'}).reset_index()
    df_hist_cat.set_index('period', inplace=True)
    df_hist_cat = df_hist_cat.sort_index()
    df_hist_cat = df_hist_cat.asfreq('D').fillna(method='ffill')

    if df_hist_cat['sales'].count() < period:
        print(f'Недостаточно данных для категории {category}. Пропускаем.')
        return df

    hist_norm_factor = df_hist_cat['sales'].max()
    df_hist_cat['sales_normalized'] = df_hist_cat['sales'] / hist_norm_factor

    trend_hist, seasonal_hist, resid_hist = multiplicative_stl(df_hist_cat['sales_normalized'], period=period)
    df_hist_cat['seasonal_factor'] = seasonal_hist

    # Сохраняем сезонные коэффициенты по датам для данной категории
    seasonal_factors_by_date = df_hist_cat[['seasonal_factor']].copy()
    seasonal_factors_historical[category] = seasonal_factors_by_date

    # Обработка прогнозных данных для каждого артикула в категории
    df_pred_cat = df[df['category'] == category].copy()
    if df_pred_cat.empty:
        print(f'Нет данных предсказаний для категории {category}. Пропускаем.')
        return df

    adjusted_articles = []

    for article in df_pred_cat['art'].unique():
        df_article = df_pred_cat[df_pred_cat['art'] == article].copy()
        df_article = df_article.sort_values('date')
        df_article.set_index('date', inplace=True)

        if len(df_article) < period:
            print(f'Недостаточно данных для артикула {article}. Пропускаем.')
            continue

        # Нормализация прогноза артикула
        pred_norm_factor = df_article['orders'].max()
        df_article['orders_normalized'] = df_article['orders'] / pred_norm_factor

        try:
            trend_pred, seasonal_pred, resid_pred = multiplicative_stl(df_article['orders_normalized'], period=period)
        except Exception as e:
            print(f'Ошибка декомпозиции для артикула {article}: {e}')
            continue

        # Десезонизация ряда артикула
        df_article['deseasonalized'] = df_article['orders_normalized'] / seasonal_pred

        # Подставляем агрегированную сезонную компоненту из исторических данных
        df_article['historical_seasonal_factor'] = df_article.index.map(
            lambda d: seasonal_factors_by_date['seasonal_factor'].get(d.replace(year=2024), np.nan)
        )
        df_article['historical_seasonal_factor'].fillna(method='ffill', inplace=True)
        df_article['historical_seasonal_factor'].fillna(method='bfill', inplace=True)

        # Воссоздаем прогноз с новой сезонной компонентой
        df_article['adjusted_orders_normalized'] = (
            df_article['deseasonalized'] * df_article['historical_seasonal_factor']
        )
        df_article['adjusted_orders'] = df_article['adjusted_orders_normalized'] * pred_norm_factor

        df_article['art'] = article
        df_article.reset_index(inplace=True)
        adjusted_articles.append(df_article[['art', 'date', 'adjusted_orders']])

    if adjusted_articles:
        adjusted_df = pd.concat(adjusted_articles, ignore_index=True)
        for _, row in adjusted_df.iterrows():
            mask = (df['art'] == row['art']) & (df['date'] == row['date'])
            df.loc[mask, 'adjusted_orders'] = row['adjusted_orders']

    return df


# Главная функция
def main_wb():
    df_s, df = load_data()
    common_categories, df_s, df = preprocess_data(df_s, df)

    seasonal_factors_historical = {}

    for category in common_categories:
        df = process_category(category, df_s, df, seasonal_factors_historical, period=7)

    # Если скорректированные заказы отсутствуют, используем исходные прогнозы
    df['adjusted_orders'] = df['adjusted_orders'].fillna(df['orders'])

    # pml_sn = PredictMLPathDataSEASON()
    final_path = 'WB_season_coef_predict.csv'

    # Сохраняем файл
    df.to_csv(final_path, index=False)


