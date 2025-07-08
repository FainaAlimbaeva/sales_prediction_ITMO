import os
from datetime import datetime, timedelta
from math import sqrt

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor, Pool
from category_encoders import TargetEncoder
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


def split_data(full_df, last_date) -> tuple:
    """Разделяем full_df на train, val (последние 31 день из train) и test (следующий день)."""
    full_df = full_df.sort_values(by=['date'])
    test_start_date = last_date + pd.Timedelta(days=1)

    train_df = full_df[full_df['date'] < test_start_date]
    test_df = full_df[full_df['date'] >= test_start_date]

    validation_start_date = pd.to_datetime(test_start_date) - pd.Timedelta(days=31)
    val_df = train_df[train_df['date'] >= validation_start_date]
    train_df = train_df[train_df['date'] < validation_start_date]

    train_df = train_df.sort_values(by='date')
    val_df = val_df.sort_values(by='date')
    test_df = test_df.sort_values(by='date')

    print('Размер тренировочного набора:', len(train_df))
    print('Размер валидационного набора:', len(val_df))
    print('Размер тестового набора:', len(test_df))
    return train_df, val_df, test_df


def encode_features(train_df, val_df):
    """Кодируем art и category через TargetEncoder для train и val."""
    encoder_art = TargetEncoder(smoothing=500)
    encoder_cat = TargetEncoder(smoothing=12)

    train_df['art_encoded'] = encoder_art.fit_transform(train_df['art'], train_df['orders'])
    val_df['art_encoded'] = encoder_art.transform(val_df['art'])

    train_df['category_encoded'] = encoder_cat.fit_transform(train_df['category'], train_df['orders'])
    val_df['category_encoded'] = encoder_cat.transform(val_df['category'])

    return train_df, val_df, encoder_art, encoder_cat


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def optimize_hyperparameters(X_train, y_train, random_seed):
    """Ищем лучшие гиперпараметры CatBoost через Optuna, опираясь на TimeSeriesSplit."""

    def objective(trial):
        param = {
            'iterations': trial.suggest_int('iterations', 1000, 2000),
            'depth': trial.suggest_int('depth', 8, 13),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.4),
            'random_strength': trial.suggest_int('random_strength', 0, 100),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.01, 10.0),
            'verbose': False,
            'loss_function': 'RMSE',
            'random_seed': random_seed,
            'thread_count': 6,
        }

        model = CatBoostRegressor(**param)
        n_splits = 3
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []

        for train_index, test_index in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_index], X_train.iloc[test_index]
            y_tr, y_val = y_train.iloc[train_index], y_train.iloc[test_index]

            train_pool_cv = Pool(data=X_tr, label=y_tr)
            val_pool_cv = Pool(data=X_val, label=y_val)
            model.fit(
                train_pool_cv, eval_set=val_pool_cv, early_stopping_rounds=150, use_best_model=True, verbose=False
            )
            preds = model.predict(val_pool_cv)
            rmse = root_mean_squared_error(y_val, preds)
            scores.append(rmse)

        return np.mean(scores)

    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=3)

    print('Лучшие параметры:', study.best_params)
    print('Лучшая метрика:', study.best_value)
    return study.best_params


def train_model_with_best_params(X_train, y_train, X_val, y_val, best_params, random_seed):
    """
    Обучаем CatBoost на (train + val) с найденными параметрами (без eval_set).
    """
    # Объединяем train и val
    X_full = pd.concat([X_train, X_val], axis=0)
    y_full = pd.concat([y_train, y_val], axis=0)

    # Создаём модель с лучшими параметрами
    model = CatBoostRegressor(
        **best_params,
        # task_type='CPU',
        # devices='0:1',
        loss_function='RMSE',
        random_seed=random_seed,
        verbose=100,
        thread_count=6,
    )

    # Обучаем на всех данных без eval_set и без ранней остановки
    model.fit(X_full, y_full)

    return model


def run_iterative_forecasting(full_train_df, test_df, generate_features_func, best_params, random_seed, n_months=4):
    """
    Итеративно предсказываем на n_months вперёд:
    - каждый месяц (итерация) обучаем модель на всех имеющихся данных (full_train_df + предыдущий test),
    - получаем новый test (следующий месяц),
    - присваиваем 'orders' = 'predictions' в тесте, чтобы далее учитывать их как "новые фактические" данные.
    На каждом шаге заново обучаем TargetEncoder на fresh train, чтобы учесть все последние данные.
    """
    all_predictions = test_df.copy()

    for month in range(n_months):
        # 1. Склеиваем исторические данные (full_train_df) и текущий test
        data = pd.concat([full_train_df, test_df], ignore_index=True)
        data['date'] = pd.to_datetime(data['date'])

        # 2. Генерация фич на склеенных данных
        last_date = data['date'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=31, freq='D')
        future_df = pd.DataFrame({'date': future_dates})
        full_df = data.groupby('art', group_keys=False).apply(generate_features_func).reset_index(drop=True)

        # 3. Разделяем full_df на train и test для текущей итерации
        test_start_date = last_date + pd.Timedelta(days=1)
        train_df_ = full_df[full_df['date'] < test_start_date].copy()
        test_df_ = full_df[full_df['date'] >= test_start_date].copy()

        # 4. Выделяем валидацию (последние 75 дней из train)
        validation_start_date = test_start_date - pd.Timedelta(days=75)
        val_df_ = train_df_[train_df_['date'] >= validation_start_date].copy()
        train_df_ = train_df_[train_df_['date'] < validation_start_date].copy()

        # 5. Склеиваем обратно, чтобы обучать модель на train+val
        combined_train_df_ = pd.concat([train_df_, val_df_], ignore_index=True)

        # ---- Переобучаем TargetEncoder на fresh train ----
        encoder_art = TargetEncoder(smoothing=500)
        encoder_cat = TargetEncoder(smoothing=12)

        combined_train_df_['art_encoded'] = encoder_art.fit_transform(
            combined_train_df_['art'], combined_train_df_['orders']
        )
        combined_train_df_['category_encoded'] = encoder_cat.fit_transform(
            combined_train_df_['category'], combined_train_df_['orders']
        )

        test_df_['art_encoded'] = encoder_art.transform(test_df_['art'])
        test_df_['category_encoded'] = encoder_cat.transform(test_df_['category'])

        # 6. Формируем X, y
        drop_cols = ['orders', 'date', 'nazvanie', 'art', 'category', 'predictions']
        X_full_train = combined_train_df_.drop(
            columns=[c for c in drop_cols if c in combined_train_df_.columns], errors='ignore'
        )
        y_full_train = combined_train_df_['orders']

        X_test = test_df_.drop(columns=[c for c in drop_cols if c in test_df_.columns], errors='ignore')
        model = CatBoostRegressor(
            **best_params,
            # task_type='CPU',
            # devices='0:1',
            loss_function='RMSE',
            verbose=100,
            random_seed=random_seed,
            thread_count=6,
        )

        # 7. Обучаем модель (обучаем только для первого месяца, для остальных периодов просто предсказываем)
        if month == 0:
            model.fit(X_full_train, y_full_train)
            # Сохраняем модель в файл (формат .cbm)
            model.save_model('models/catboost_model.cbm')
        else:
            model.load_model('models/catboost_model.cbm')
            
        # 8. Предсказываем
        test_predictions = model.predict(X_test)
        test_predictions = np.maximum(0, test_predictions)
        test_df_['predictions'] = test_predictions
        test_df_['orders'] = test_df_['predictions']

        # 9. Готовимся к следующему месяцу
        test_df = test_df_.copy()
        all_predictions = pd.concat([all_predictions, test_df], ignore_index=True)

        # 10. Обновляем full_train_df
        full_train_df = pd.concat([train_df_, val_df_], ignore_index=True)

    return all_predictions
