import mysql.connector

def get_connection():
    return mysql.connector.connect(
        host="140.131.114.242",
        user="smartcare_db",
        password="SmartCare114@2",
        database="114-402"
    )
