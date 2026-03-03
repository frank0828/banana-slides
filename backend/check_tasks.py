import sqlite3
conn = sqlite3.connect('instance/database.db')
cursor = conn.cursor()
cursor.execute('SELECT id, status, task_type, created_at FROM tasks ORDER BY created_at DESC LIMIT 10')
tasks = cursor.fetchall()
print(f"{'ID':<40} | {'Status':<12} | {'Type':<20} | {'Created'}")
print("-" * 100)
for row in tasks:
    print(f"{row[0]:<40} | {row[1]:<12} | {row[2]:<20} | {row[3]}")
conn.close()
