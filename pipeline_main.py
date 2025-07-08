import os
import warnings
from datetime import datetime, timedelta

from catboost_sales_prediction import *
from seasonal_corrections import *

warnings.filterwarnings('ignore')
import logging

# Включаем логирование Optuna
optuna.logging.set_verbosity(optuna.logging.INFO)

# ----------------------------------
# ---------- НАЧАЛО СКРИПТА --------
# ----------------------------------


def main():
    """Главная функция, где выполняем весь пайплайн обучения и итеративного предсказания."""

    file_path = 'predict/WB_data.csv'
    RANDOM_SEED = 777

    # 1. Загрузка и подготовка данных
    data = load_and_prepare_data(file_path)
    data = aggregate_and_filter_data(data)
    data = add_lag_and_lead_features(data)

    # 2. Генерация признаков (прогноз на 31 день вперёд для каждого артикула)
    full_df = generate_future_features(data)
    last_date = data['date'].max()
    last_date_M = data['date'].max()

    # 3. Разделяем на train / val / test
    train_df, val_df, test_df = split_data(full_df, last_date)

    # 4. Кодируем train и val
    train_df, val_df, encoder_art, encoder_cat = encode_features(train_df, val_df)

    # 5. Готовим X, y
    X_train = train_df.drop(['orders', 'date' 'art', 'category'], axis=1)
    y_train = train_df['orders']
    X_val = val_df.drop(['orders', 'date', 'art', 'category'], axis=1)
    y_val = val_df['orders']

    # 6. Ищем лучшие гиперпараметры
    best_params = optimize_hyperparameters(X_train, y_train, RANDOM_SEED)

    # 7. Обучаем модель на train+val
    model = train_model_with_best_params(X_train, y_train, X_val, y_val, best_params, RANDOM_SEED)

    # 8. Первичное предсказание на test (текущий месяц)
    test_df['art_encoded'] = encoder_art.transform(test_df['art'])
    test_df['category_encoded'] = encoder_cat.transform(test_df['category'])

    X_test = test_df.drop(['orders', 'date', 'art', 'category'], axis=1)
    test_predictions = model.predict(X_test)
    test_predictions = np.maximum(0, test_predictions)
    test_df['predictions'] = test_predictions
    test_df['orders'] = test_predictions

    print('Первичные предсказания на тестовой выборке получены.')

    # 9. Формируем full_train_df (train + val)
    full_train_df = pd.concat([train_df, val_df], ignore_index=True)

    # 10. Запускаем итеративный прогноз (например, на 2 месяца)
    all_predictions = run_iterative_forecasting(
        full_train_df=full_train_df,
        test_df=test_df,
        generate_features_func=generate_future_features,
        best_params=best_params,
        random_seed=RANDOM_SEED,
        n_months=12,
    )

    print('Итеративные предсказания завершены.')
    all_predictions['date'] = pd.to_datetime(all_predictions['date'])
    all_predictions['orders'] = all_predictions['predictions'] * 1.30  # подменяем 'orders'

    # Сдвигаем дату (примерно на 31 день назад)
    all_predictions['date'] = all_predictions['date'] - pd.DateOffset(days=31)
    # или отфильтровываем
    start_date = last_date_M
    end_date = '2025-12-31'
    all_predictions = all_predictions[(all_predictions['date'] >= start_date) & (all_predictions['date'] <= end_date)]

    columns = [ 'date', 'orders', 'art', 'category']
    final_df = all_predictions[[c for c in columns if c in all_predictions.columns]]

    final_path = 'predict/WB_data.csv'
    # Сохраняем файл
    final_df.to_csv(final_path, index=False)

    print(f'Файл успешно сохранён по пути: {final_path}')
    print(final_df.head(10))


# ----------------------------------
# ---------- КОНЕЦ СКРИПТА ---------
# ----------------------------------


def run_full_script():
    result = main()
