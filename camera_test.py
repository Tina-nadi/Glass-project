import easyocr

reader = easyocr.Reader(['fa'])  # فارسی
result = reader.readtext('test.jpg')
print(result)
