# Данный проект для JMLC ITMO
# Прогноз спроса на товары на маркетплейсах

Проект реализовывался для использования в дагах в Airflow.


- notebooks/Выгрузка данных с MpStats категории.ipynb - ноутбук по выгрузке категорий ниш по их названиям, с использованием API MPstats
- notebooks/Кластеризация категорий DTW.ipynb - ноутбук для разбиение категорий ниш на кластеры, с применением библиотеки tslearn
- utils/db.py - класс для подключения к базе данных
- data_preparation.py - модуль для подготовки данных и генерации фич для прогноза спроса
- catboost_sales_prediction.py - модуль для прогноза спроса товаров с помощью CatboostRegressor и optuna
- seasonal_corrections.py - модуль для корректировки прогноза сезонной компонентой по категориям
- pipeline_main.py - пайплайн прогноза спроса.

Использовано окружение на python-3.9.23 

requirements.txt содержит необходимые библиотеки