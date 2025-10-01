import os
import urllib.request

# مسیر پیش‌فرض مدل‌های EasyOCR
EASYOCR_MODEL_DIR = os.path.join(os.path.expanduser("~"), ".EasyOCR", "model")
os.makedirs(EASYOCR_MODEL_DIR, exist_ok=True)

# لیست مدل‌های لازم
models = {
    "craft_mlt_25k.pth": "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/craft_mlt_25k.pth",
    "fa.pth": "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/fa.pth",
    "en.pth": "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/en.pth"
}


def download_with_resume(url, filepath):
    """دانلود با قابلیت ادامه در صورت قطع شدن"""
    temp_file = filepath + ".part"
    file_mode = "ab" if os.path.exists(temp_file) else "wb"
    downloaded = os.path.getsize(temp_file) if os.path.exists(temp_file) else 0

    req = urllib.request.Request(url)
    if downloaded > 0:
        req.add_header("Range", f"bytes={downloaded}-")

    with urllib.request.urlopen(req) as response, open(temp_file, file_mode) as out_file:
        total_size = int(response.info().get("Content-Length", -1)) + downloaded
        block_size = 8192
        while True:
            buffer = response.read(block_size)
            if not buffer:
                break
            out_file.write(buffer)
            downloaded += len(buffer)
            done = int(50 * downloaded / total_size)
            print(f"\r[{'=' * done}{' ' * (50 - done)}] {downloaded / 1e6:.1f}/{total_size / 1e6:.1f} MB", end="")

    os.rename(temp_file, filepath)
    print(f"\n✅ دانلود کامل شد: {filepath}")


def check_and_download_models():
    for filename, url in models.items():
        filepath = os.path.join(EASYOCR_MODEL_DIR, filename)
        if not os.path.exists(filepath):
            print(f"⬇️ در حال دانلود مدل: {filename}")
            download_with_resume(url, filepath)
        else:
            print(f"✅ مدل {filename} از قبل وجود دارد")


if __name__ == "__main__":
    print(f"📂 مسیر مدل‌ها: {EASYOCR_MODEL_DIR}")
    check_and_download_models()
