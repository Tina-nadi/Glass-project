import cv2
import sqlite3
import time
from ultralytics import YOLO

# بارگذاری مدل
model = YOLO("yolov8n.pt")

# اتصال به دیتابیس
conn = sqlite3.connect("detections.db")
cursor = conn.cursor()

# ایجاد جدول اگر وجود ندارد
cursor.execute('''
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT,
    confidence REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')
conn.commit()

# آخرین زمان ثبت هر object
last_seen = {}

def run_object_detection(frame):
    """
    پردازش فریم با YOLOv8 و ذخیره اشیا در دیتابیس
    ورودی: frame (BGR numpy array)
    خروجی: فریم آنوتیت شده (annotated frame)
    """
    results = model(frame)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])

            # زمان فعلی
            now = time.time()

            # اگر object جدید باشه یا بیشتر از 2 ثانیه از آخرین ثبتش گذشته باشه → ذخیره کن
            if label not in last_seen or (now - last_seen[label]) > 2:
                cursor.execute(
                    "INSERT INTO detections (label, confidence) VALUES (?, ?)",
                    (label, conf)
                )
                conn.commit()
                last_seen[label] = now
                print(f"Saved: {label} ({conf:.2f})")

    # برگردوندن فریم آنوتیت شده
    annotated_frame = results[0].plot()
    return annotated_frame

# اگر مستقیم اجرا شد، حالت standalone
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame = run_object_detection(frame)
        cv2.imshow("YOLOv8 Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    conn.close()
    cv2.destroyAllWindows()
