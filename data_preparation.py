import os
from datetime import datetime, timedelta

import holidays
import numpy as np
import pandas as pd

def load_and_prepare_data(file_path) -> pd.DataFrame:
    """Загружаем CSV, превращаем date в datetime и сортируем по art и date."""
    data: pd.DataFrame = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(by=['art', 'date'])
    return data


def filter_sales(data):
    """Фильтруем строки, начиная с первой продажи (orders>0)."""
    first_sale_idx = data[data['orders'] > 0].index.min()
    return data.loc[first_sale_idx:]


def aggregate_and_filter_data(data):
    """Группируем по art и date, суммируем orders, берём последнее название и категорию."""
    filtered_df = data.groupby('art', group_keys=False).apply(filter_sales)
    filtered_df = filtered_df.groupby(['art', 'date'], as_index=False).agg(
        {
            'orders': 'sum', 
             # 'nazvanie': 'last', 
             'category': 'last'
        }
    )
    return filtered_df


def add_lag_and_lead_features(data):
    """Добавляем лаги, опережающие признаки и заменяем выбросы средним по 4 дням."""
    data['lag_2'] = data.groupby('art')['orders'].shift(2)
    data['lag_1'] = data.groupby('art')['orders'].shift(1)
    data['lead_1'] = data.groupby('art')['orders'].shift(-1)
    data['lead_2'] = data.groupby('art')['orders'].shift(-2)

    data['mean_4_days'] = data[['lag_2', 'lag_1', 'lead_1', 'lead_2']].mean(axis=1)

    data['orders'] = data.apply(
        lambda row: row['mean_4_days'] if row['orders'] > 40 and pd.notnull(row['mean_4_days']) else row['orders'],
        axis=1,
    )
    data.drop(columns=['lag_2', 'lag_1', 'lead_1', 'lead_2', 'mean_4_days'], inplace=True)
    return data


def generate_future_features(data):
    """Формируем будущие даты и создаём временные/сезонные признаки."""
    last_date = data['date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=31, freq='D')
    future_df = pd.DataFrame({'date': future_dates})

    def generate_features(group):
        group = group.sort_values(by='date').reset_index(drop=True)
        future_group = future_df.copy()
        future_group['art'] = group['art'].iloc[0]
        # future_group['nazvanie'] = group['nazvanie'].iloc[0]

        combined_df = pd.concat([group, future_group], ignore_index=True)
        combined_df[['category', 'art']] = combined_df[['category', 'art']].fillna(
            method='ffill'
        )

        # Признаки дат
        combined_df['day_of_week'] = combined_df['date'].dt.dayofweek
        combined_df['month'] = combined_df['date'].dt.month
        combined_df['year'] = combined_df['date'].dt.year
        combined_df['quarter'] = combined_df['date'].dt.quarter
        combined_df['day_of_month'] = combined_df['date'].dt.day
        combined_df['day_of_year'] = combined_df['date'].dt.dayofyear
        combined_df['week_of_year'] = combined_df['date'].dt.isocalendar().week
        combined_df['is_weekend'] = (combined_df['day_of_week'] >= 5).astype(int)
        combined_df['week_of_month'] = combined_df['date'].dt.day // 7 + 1

        # Праздники
        ru_holidays = holidays.RU()
        combined_df['is_holiday'] = combined_df['date'].apply(lambda x: x in ru_holidays).astype(int)

        # Гармонические и лог-признаки
        periods = [365, 180, 90, 60, 28, 14]
        harmonics = [1, 2, 3]
        for period in periods:
            for harmonic in harmonics:
                combined_df[f'sin_{period}_{harmonic}'] = np.sin(
                    2 * np.pi * harmonic * combined_df['date'].dt.dayofyear / period
                )
                combined_df[f'cos_{period}_{harmonic}'] = np.cos(
                    2 * np.pi * harmonic * combined_df['date'].dt.dayofyear / period
                )

                phase_shift = 28
                combined_df[f'sin_{period}_{harmonic}_shift'] = np.sin(
                    2 * np.pi * harmonic * (combined_df['date'].dt.dayofyear + phase_shift) / period
                )
                combined_df[f'cos_{period}_{harmonic}_shift'] = np.cos(
                    2 * np.pi * harmonic * (combined_df['date'].dt.dayofyear + phase_shift) / period
                )

                combined_df[f'log_sin_{period}_{harmonic}'] = np.log(
                    np.abs(combined_df[f'sin_{period}_{harmonic}']) + 1
                )
                combined_df[f'log_cos_{period}_{harmonic}'] = np.log(
                    np.abs(combined_df[f'cos_{period}_{harmonic}']) + 1
                )

        # Скользящие средние, лаги, сглаживание и другие статистики
        moving_averages = [31, 60, 90]
        for window in moving_averages:
            combined_df[f'weighted_ma_{window}'] = (
                combined_df['orders']
                .shift(1)
                .rolling(window=window)
                .apply(lambda x: np.average(x, weights=np.linspace(1, 2, num=len(x))))
            )

        lags = [31, 60, 90]
        for lag in lags:
            combined_df[f'lag_{lag}'] = combined_df['orders'].shift(lag)

        for window in moving_averages:
            combined_df[f'ma_{window}'] = combined_df['orders'].shift(1).rolling(window=window).mean()

        best_alpha = 0.01
        best_span = 5
        best_halflife = 1
        best_com = 0.5
        shift_values = [31]
        for shift in shift_values:
            combined_df[f'ewm_alpha_{best_alpha}_shift_{shift}'] = (
                combined_df['orders'].shift(shift).ewm(alpha=best_alpha).mean()
            )
            combined_df[f'ewm_span_{best_span}_shift_{shift}'] = (
                combined_df['orders'].shift(shift).ewm(span=best_span).mean()
            )
            combined_df[f'ewm_halflife_{best_halflife}_shift_{shift}'] = (
                combined_df['orders'].shift(shift).ewm(halflife=best_halflife).mean()
            )
            combined_df[f'ewm_com_{best_com}_shift_{shift}'] = (
                combined_df['orders'].shift(shift).ewm(com=best_com).mean()
            )

        for window in moving_averages:
            combined_df[f'std_{window}'] = combined_df['orders'].shift(1).rolling(window=window).std()
            combined_df[f'max_{window}'] = combined_df['orders'].shift(1).rolling(window=window).max()
            combined_df[f'min_{window}'] = combined_df['orders'].shift(1).rolling(window=window).min()
            combined_df[f'mean_{window}'] = combined_df['orders'].shift(1).rolling(window=window).mean()
            combined_df[f'quantile_25_{window}'] = combined_df['orders'].shift(1).rolling(window=window).quantile(0.25)
            combined_df[f'quantile_75_{window}'] = combined_df['orders'].shift(1).rolling(window=window).quantile(0.75)
            combined_df[f'sum_{window}'] = combined_df['orders'].shift(1).rolling(window=window).sum()
            combined_df[f'range_{window}'] = combined_df[f'max_{window}'] - combined_df[f'min_{window}']
            combined_df[f'median_{window}'] = combined_df['orders'].shift(1).rolling(window=window).median()
            combined_df[f'cv_{window}'] = combined_df[f'std_{window}'] / combined_df[f'mean_{window}']
            combined_df[f'variance_{window}'] = combined_df['orders'].shift(1).rolling(window=window).var()
            combined_df[f'cumsum_{window}'] = combined_df['orders'].shift(1).rolling(window=window).sum()
            combined_df[f'cumprod_{window}'] = combined_df['orders'].shift(1).rolling(window=window).apply(np.prod)
            combined_df[f'trend_{window}'] = (
                combined_df['orders']
                .shift(1)
                .rolling(window=window)
                .apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if np.all(~np.isnan(x)) else np.nan, raw=True)
            )

        for lag in lags:
            for window in moving_averages:
                combined_df[f'corr_orders_lag{lag}_{window}'] = (
                    combined_df['orders'].shift(1).rolling(window=window).corr(combined_df[f'lag_{lag}'].shift(1))
                )
                combined_df[f'cov_orders_lag{lag}_{window}'] = (
                    combined_df['orders'].shift(1).rolling(window=window).cov(combined_df[f'lag_{lag}'].shift(1))
                )

        return combined_df

    full_df = data.groupby('art', group_keys=False).apply(generate_features).reset_index(drop=True)
    return full_df
