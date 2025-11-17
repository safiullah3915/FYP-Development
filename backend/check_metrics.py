import sqlite3

conn = sqlite3.connect('db.sqlite3')
cursor = conn.cursor()

cursor.execute('SELECT COUNT(*) FROM startup_trending_metrics')
metrics_count = cursor.fetchone()[0]
print(f'Trending metrics count: {metrics_count}')

cursor.execute("SELECT COUNT(*) FROM startups WHERE status='active'")
active_startups = cursor.fetchone()[0]
print(f'Active startups count: {active_startups}')

# Get some sample IDs to check
cursor.execute('SELECT id, title FROM startups WHERE status="active" LIMIT 15')
startups = cursor.fetchall()
print(f'\nFirst 15 active startups:')
for startup_id, title in startups:
    print(f'  - {startup_id}: {title}')

conn.close()





