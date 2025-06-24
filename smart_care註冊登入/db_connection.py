# db_connection.py

import mysql.connector
from dotenv import load_dotenv
import os

# 讀取 .env 檔案（只需要做一次）
load_dotenv()

def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host=os.getenv("MYSQL_HOST"),
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            database=os.getenv("MYSQL_DB")
        )
        return conn
    except mysql.connector.Error as e:
        print(f" 無法連接資料庫：{e}")
        raise
