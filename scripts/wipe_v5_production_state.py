import sqlite3
try:
    conn = sqlite3.connect('/app/data/trading.db')
    conn.execute('DELETE FROM v5_state')
    conn.commit()
    conn.close()
    print('V5 State Wiped Successfully.')
except Exception as e:
    print(f'Error: {e}')
