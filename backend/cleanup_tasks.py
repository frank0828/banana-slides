import sqlite3
from datetime import datetime, timedelta

conn = sqlite3.connect('instance/database.db')
cursor = conn.cursor()

# 找出超过10分钟还在PROCESSING的任务
cutoff_time = (datetime.now() - timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M:%S')

cursor.execute('''
    SELECT id, task_type, created_at 
    FROM tasks 
    WHERE status = 'PROCESSING' 
    AND created_at < ?
''', (cutoff_time,))

stuck_tasks = cursor.fetchall()

if stuck_tasks:
    print(f"Found {len(stuck_tasks)} stuck tasks:")
    for task in stuck_tasks:
        print(f"  - {task[0][:12]}... | {task[1]} | {task[2]}")
    
    # 更新为FAILED
    task_ids = [task[0] for task in stuck_tasks]
    placeholders = ','.join(['?'] * len(task_ids))
    cursor.execute(f'''
        UPDATE tasks 
        SET status = 'FAILED', 
            error_message = 'Task timeout - auto-failed by cleanup script'
        WHERE id IN ({placeholders})
    ''', task_ids)
    
    conn.commit()
    print(f"\n✅ Successfully marked {len(stuck_tasks)} tasks as FAILED")
else:
    print("✅ No stuck tasks found")

conn.close()
