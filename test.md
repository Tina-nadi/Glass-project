



```py
import pyttsx3

engine = pyttsx3.init()

engine.setProperty('voice', 'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\MSTTS_V110_faIR_Hoda')
engine.setProperty('rate', 150)

def text_to_speech(text):
    engine.say(text)
    engine.runAndWait()

```