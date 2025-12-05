import cv2
import time
import mysql.connector
import os
import threading
import pyttsx3

engine = pyttsx3.init()

engine.setProperty('voice', 'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\MSTTS_V110_faIR_Hoda')
engine.setProperty('rate', 150)
import easyocr
from ultralytics import YOLO

# --- مسیر فایل‌ها ---
YOLO_MODEL_PATH = "yolov8n.pt"  # مدل YOLOv8
TEMP_IMAGE_PATH = "temp_frame.jpg"
TEMP_AUDIO_PATH = "temp_audio.mp3"

# اتصال به دیتابیس
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="Glass-project"
)
cursor = conn.cursor()


# --- بارگذاری مدل YOLOv8 ---
model = YOLO(YOLO_MODEL_PATH)

# --- OCR و TTS ---
reader = easyocr.Reader(['fa', 'en'])
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
                    "INSERT INTO Data (detect_obj_name, time) VALUES (%s, %s)",
                    (label, int(time.time()))
                )
                conn.commit()  # اجرای کوئری
                last_seen[label] = now
                print(f"Saved: {label} ({conf:.2f})")

    annotated_frame = results[0].plot()
    return annotated_frame

def text_to_speech(text):
    engine.say(text)
    engine.runAndWait()

def run_ocr_and_tts(image_path):
    global processing_temp
    img = cv2.imread(image_path)

    # OCR فارسی + انگلیسی
    result = reader.readtext(img, detail=1, paragraph=True)

    text = " ".join([res[1] for res in result])
    if not text.strip():
        text = "⚠️ متنی برای خواندن پیدا نشد!"

    cursor.execute(
        "INSERT INTO DataText (detected_text, time) VALUES (%s, %s)",
        (text.encode("utf-8"), int(time.time()))
    )
    conn.commit()

    print(f"📝 متن استخراج‌شده:\n{text}\n")

    processing_temp = True
    text_to_speech(text)


def main():
    global processing_temp, stop_audio_flag
    cap = cv2.VideoCapture(0)
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
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
