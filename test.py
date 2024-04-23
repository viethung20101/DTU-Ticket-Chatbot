from config import *

with psycopg2.connect(CONN_STR) as conn:
    data = pd.read_sql("SELECT * FROM tickets", conn)
print(data)