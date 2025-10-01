import cv2
import sqlite3
import time
import os
import threading
import pyttsx3
import easyocr
from ultralytics import YOLO

# --- مسیر فایل‌ها ---
YOLO_MODEL_PATH = "yolov8n.pt"  # مدل YOLOv8
DB_PATH = "detections.db"       # دیتابیس
TEMP_IMAGE_PATH = "temp_frame.jpg"
TEMP_AUDIO_PATH = "temp_audio.mp3"

# --- اتصال به دیتابیس ---
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT,
    confidence REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')
conn.commit()

# --- بارگذاری مدل YOLOv8 ---
model = YOLO(YOLO_MODEL_PATH)

# --- OCR و TTS ---
reader = easyocr.Reader(['en'])  # فارسی اگر بخوای اضافه کنی: ['fa','en']
engine = pyttsx3.init()

# --- مدیریت آخرین زمان ثبت اشیا ---
last_seen = {}

# --- وضعیت برنامه ---
processing_temp = False  # آیا OCR+TTS در حال اجراست
stop_audio_flag = False

def run_object_detection(frame):
    global last_seen
    results = model(frame)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])
            now = time.time()

            if label not in last_seen or (now - last_seen[label]) > 2:
                cursor.execute(
                    "INSERT INTO detections (label, confidence) VALUES (?, ?)",
                    (label, conf)
                )
                conn.commit()
                last_seen[label] = now
                print(f"Saved: {label} ({conf:.2f})")

    annotated_frame = results[0].plot()
    return annotated_frame

def text_to_speech(text):
    global stop_audio_flag
    stop_audio_flag = False
    engine.save_to_file(text, TEMP_AUDIO_PATH)
    engine.runAndWait()
    # پخش صوت
    os.system(f'start /min wmplayer "{TEMP_AUDIO_PATH}"')  # برای ویندوز

def run_ocr_and_tts(image_path):
    global processing_temp
    img = cv2.imread(image_path)
    result = reader.readtext(img)
    text = " ".join([res[1] for res in result])
    if not text.strip():
        text = "⚠️ متنی برای خواندن پیدا نشد!"
    print(f"📝 متن استخراج‌شده:\n{text}\n")
    processing_temp = True
    text_to_speech(text)

def main():
    global processing_temp, stop_audio_flag
    cap = cv2.VideoCapture(0)
    temp_frame_saved = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # فقط وقتی که OCR+TTS فعال نیست → YOLOv8 اجرا میشه
        if not processing_temp:
            annotated_frame = run_object_detection(frame)
        else:
            annotated_frame = frame.copy()

        cv2.imshow("YOLOv8 + OCR", annotated_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):  # اسپیس → ذخیره فریم و OCR
            if not processing_temp:
                cv2.imwrite(TEMP_IMAGE_PATH, frame)
                print("📷 عکس گرفته شد!")
                threading.Thread(target=run_ocr_and_tts, args=(TEMP_IMAGE_PATH,), daemon=True).start()
            else:
                # پاک کردن فایل موقت و ادامه YOLO
                if os.path.exists(TEMP_IMAGE_PATH):
                    os.remove(TEMP_IMAGE_PATH)
                if os.path.exists(TEMP_AUDIO_PATH):
                    os.remove(TEMP_AUDIO_PATH)
                print("🔄 فایل‌های موقت حذف شدند. پردازش YOLO ادامه دارد.")
                processing_temp = False

    cap.release()
    conn.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
