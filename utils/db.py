from creds import DB_HOST, DB_USER, DB_PASW, DB_DATABASE 

# -*- coding: utf-8 -*-
"""
Database class
"""
import logging
from typing import Any, Dict, List
from pandas import Timestamp
import datetime
# import mysql.connector
import psycopg2
import pandas as pd

# pip install mysql-connector-python
# from database_connector.settings import *

class MsDatabase:
    def __init__(self,
                 num_rows_to_load=1000,
                 # max_allowed_packet=33554432,  # 'SELECT @@max_allowed_packet; SHOW VARIABLES LIKE "max_allowed_packet"'
                 host=None,
                 user=None,
                 passwd=None,
                 database=None,
                 wait_timeout=10):
        self.user = user
        self.host = host
        self.passwd = passwd
        self.database = database
        self.conn = None
        self.wait_timeout = wait_timeout
        self.cur = None
        self.num_rows_to_load = num_rows_to_load
        # self.max_allowed_packet = max_allowed_packet
        self.conn = psycopg2.connect(
            dbname=database,
            host=host,
            user=user,
            password=passwd,
        )
        self.cur = self.conn.cursor()
        self.cur.execute(f"SET statement_timeout TO '{self.wait_timeout}min'")

    def reconnect(self):
        self.conn.close()
        self.cur.close()
        self.conn = psycopg2.connect(
            dbname=self.database,
            host=self.host,
            user=self.user,
            password=self.passwd,
        )
        self.cur = self.conn.cursor()
        self.cur.execute(f"SET statement_timeout TO '{self.wait_timeout}min'")
        logging.info('PostgreSQL cursor reopened')

    def create_table_from_dict_template(
            self,
            schema: str,
            table: str,
            dict_template: Dict[str, str],
            index_field=None
    ) -> None:

        fields_text = ''
        # self.check_and_correct_column_names(dict_template.keys())
        for key in dict_template:
            fields_text += f'{key} {dict_template[key]},'
        fields_text = fields_text[:-1]
        if index_field:
            req = f"CREATE TABLE IF NOT EXISTS {schema}.{table} ({fields_text} PRIMARY KEY ({index_field}))"
        else:
            req = f"CREATE TABLE IF NOT EXISTS {schema}.{table} ({fields_text})"
        try:
            self.cur.execute(req)
            self.conn.commit()
        except Exception as e:
            logging.warning(f'PostgreSQL create table {table} error')
            logging.warning(str(e))
            logging.warning(req)
            return False
        return True

    def check_string(self, i: Any) -> Any:
        if isinstance(i, str):
            i = i.replace('[', '').replace(']', '').replace("'", "''").replace('"', "''")

        if i == '':
            return "NULL"
        else:
            return str(i)

    # Создание SQL-запроса на заполнение таблицы в БД данными датафрейма
    def insert_values(self, df: pd.DataFrame, schema: str, table_name: str) -> None:
        data = df.fillna('')
        cols = str(tuple(self.check_and_correct_column_names(data.columns))).replace("'", "")

        try:
        # if np.ceil(data.memory_usage().sum() / self.max_allowed_packet) > 1:# проверка должна быть на объем памяти используемый
        #     steps = np.ceil(data.memory_usage().sum() / self.max_allowed_packet)

            steps = int(data.shape[0] // self.num_rows_to_load)
            for step in range(steps + 1):

                l = step * self.num_rows_to_load
                r = min(l + self.num_rows_to_load + 1, data.shape[0])

                if step > 0:  # добавляем 1 чтобы не дублировались данные
                    l = step * self.num_rows_to_load + 1

                val = (
                    str([tuple([self.check_string(i) for i in x]) for x in (data.iloc[l:r].to_numpy())])[1:-1]
                   .replace('"', "'")
                   .replace("'NULL'", "NULL")
                )

                req = f"INSERT INTO {schema}.{table_name} {cols} VALUES {val};"

                self.cur.execute(req)
                self.conn.commit()
                logging.info(f'data {l}:{r} is loaded')

                # else:
                #     val = str([tuple([func(i) for i in x]) for x in (data.to_numpy())])[1:-1].replace('"','')
                #     req = f"INSERT INTO {table_name} {cols} VALUES {val};"
                #     self.cur.execute(req)
                #     self.conn.commit()
        except Exception as e:
            logging.warning(f'PostgreSQL insert values {schema}.{table_name} error, data shape is {data.shape}')
            logging.warning(str(e))
            return False

    
    def truncate_and_insert_values(self, df: pd.DataFrame, table_schema: str, table_name: str) -> None:
        data = df.fillna('')
        # cols = str(tuple(self.check_and_correct_column_names(data.columns))).replace("'", "")
        self.cur.execute(f"""SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = '{table_schema}' AND table_name = '{table_name}'
            ORDER BY ordinal_position ASC;""")

        cols = self.cur.fetchall()
        cols = str(tuple([name[0] for name in cols])).replace("'", "")
        try:
        # if np.ceil(data.memory_usage().sum() / self.max_allowed_packet) > 1:# проверка должна быть на объем памяти используемый
        #     steps = np.ceil(data.memory_usage().sum() / self.max_allowed_packet)
            req1 = f"TRUNCATE TABLE {table_schema}.{table_name};"
            self.cur.execute(req1)
            logging.info(f'table {table_schema}.{table_name} is truncated')

            steps = int(data.shape[0] // self.num_rows_to_load)
            for step in range(steps + 1):

                l = step * self.num_rows_to_load
                r = min(l + self.num_rows_to_load + 1, data.shape[0])

                if step > 0:  # добавляем 1 чтобы не дублировались данные
                    l = step * self.num_rows_to_load + 1

                val = (
                    str([tuple([self.check_string(i) for i in x]) for x in (data.iloc[l:r].to_numpy())])[1:-1]
                   .replace('"', "'")
                   .replace("'NULL'", "NULL")
                )
                print(cols)
    
                req2 = f"INSERT INTO {table_schema}.{table_name} {cols} VALUES {val};"

                self.cur.execute(req2)
                logging.info(f'data {l}:{r} is loaded')

            self.conn.commit()

                # else:
                #     val = str([tuple([func(i) for i in x]) for x in (data.to_numpy())])[1:-1].replace('"','')
                #     req = f"INSERT INTO {table_name} {cols} VALUES {val};"
                #     self.cur.execute(req)
                #     self.conn.commit()
        except Exception as e:
            logging.warning(f'PostgreSQL insert values {table_schema}.{table_name} error, data shape is {data.shape}')
            logging.warning(str(e))
            return False

    # Загрузка таблицы
    def get_table(self, table_schema: str, table_name: str, columns = True) -> pd.DataFrame:
        try:
            req = f"SELECT * FROM {table_schema}.{table_name}"
            self.cur.execute(req)
            self.conn.commit()
            rows = self.cur.fetchall()
            df = pd.DataFrame(rows)


            if columns:
                self.cur.execute(f"""SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = '{table_schema}' AND table_name = '{table_name}'
                ORDER BY ordinal_position ASC;""")
                column_names = self.cur.fetchall()
                column_names = [name[0] for name in column_names]
                df.columns = column_names
            logging.info(f'table {table_name} is loaded')
            return df

        except Exception as e:
            logging.warning(f'PostgreSQL read table {table_name} error')
            logging.warning(str(e))
            return False

    def get_column_names(self, table_schema: str, table_name: str, columns = True):

        self.cur.execute(f"""SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = '{table_schema}' AND table_name = '{table_name}'
        ORDER BY ordinal_position ASC;""")
        column_names = self.cur.fetchall()
        # column_names = [name[0] for name in column_names]

        return column_names

    
    # Загрузка таблицы
    def get_table_by_request(self, req: str, column_names=False) -> pd.DataFrame:
        try:
            self.cur.execute(req)
            self.conn.commit()
            rows = self.cur.fetchall()
            df = pd.DataFrame(rows)
            if column_names:
                df.columns = column_names
            logging.info(f'table by request is loaded')
            return df

        except Exception as e:
            logging.warning(f'PostgreSQL read table {table_name} error')
            logging.warning(str(e))
            return False
            
    # Загрузка таблицы
    def update_table_by_request(self, req: str, table_name: str):
        try:
            self.cur.execute(req)
            self.conn.commit()
            logging.info(f'table by request is loaded')

        except Exception as e:
            logging.warning(f'PostgreSQL update table {table_name} error')
            logging.warning(str(e))
            return False
            
    # Очистка таблицы
    def truncate_table(self, table_schema: str, table_name: str) -> None:
        try:
            req = f"TRUNCATE TABLE {table_schema}.{table_name};"
            self.cur.execute(req)
            self.conn.commit()
            logging.info(f'table {table_schema}.{table_name} is truncated')
        except Exception as e:
            logging.warning(f'PostgreSQL truncate table {table_schema}.{table_name} error')
            logging.warning(str(e))
            return False

    # Удаление таблицы
    def drop_table(self, table_schema: str, table_name: str) -> None:
        try:
            req = f"DROP TABLE {table_schema}.{table_name};"
            self.cur.execute(req)
            self.conn.commit()
            logging.info(f'table {table_schema}.{table_name} is dropped')
        except Exception as e:
            logging.warning(f'PostgreSQL drop table {table_schema}.{table_name} error')
            logging.warning(str(e))
            return False

    # Загрузка списка таблиц по характерному признаку
    def get_tables_by_folder(self, folder: str) -> List[str]:
        req = f"""SELECT table_name FROM information_schema.tables
        WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
        AND table_schema IN ('ipolyakov_db', 'ir_db')
        AND table_name like '{folder}%';"""
        self.cur.execute(req)
        rows = self.cur.fetchall()
        return [row[0] for row in rows]

    # Проверка наличия пробелов в названиях колонок
    def check_and_correct_column_names(self, columns: Any) -> List[str]:
        return [col.replace(' ', '_') for col in columns]
        
    def update_table_by_art_date(
        self, 
        table_schema: str, 
        table_name: str, 
        art: str, 
        date: Timestamp, 
        columns: List[str], 
        values: List[int],
    ):
        
        set_part = ''
        for i in range(len(columns)):
            set_part += columns[i] + ' = ' + str(values[i]) + ', '
        
        set_part = set_part[:-2]
            
        req = f"""UPDATE {table_schema}.{table_name} SET {set_part} WHERE art='{art}' and date = '{date}'""";
        try:
            self.cur.execute(req)
            self.conn.commit()
            logging.info(f'table {table_schema}.{table_name} is dropped')
        except Exception as e:
            logging.warning(f'PostgreSQL update table {table_schema}.{table_name} error')
            logging.warning(str(e))
            return False
            
    # Закрыть коннект
    def close(self) -> None:
        self.conn.close()
        self.cur.close()
        logging.info('PostgreSQL cursor closed')
        return
# if __name__ == "__main__":
#     db = MsDatabase(host=MYSQL_HOST, user=MYSQL_USER, passwd=MYSQL_PASW,
#                             database=MYSQL_DATABASE)
db = MsDatabase(host=DB_HOST, user=DB_USER, passwd=DB_PASW,
                        database=DB_DATABASE)