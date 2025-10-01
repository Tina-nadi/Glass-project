import sqlite3

# اتصال به دیتابیس
conn = sqlite3.connect("detections.db")
cursor = conn.cursor()

# خوندن همه رکوردها
cursor.execute("SELECT id, label, confidence, timestamp FROM detections")
rows = cursor.fetchall()

# اگر دیتابیس خالی بود
if not rows:
    print("📂 دیتابیس خالیه! هیچ دیتایی ثبت نشده.")
else:
    print("📊 محتویات دیتابیس detections:\n")
    for row in rows:
        print(f"ID: {row[0]} | Label: {row[1]} | Confidence: {row[2]:.2f} | Time: {row[3]}")

conn.close()
