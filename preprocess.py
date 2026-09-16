import re
import string

HINDI_STOPWORDS = set("""
अंदर अत अपना अपनी अपने अभी आदि आप इत्यादि इन इनका इन्हीं इन्हें इन्होंने इस इसका
इसकी इसके इसमें इसी इसे उन उनका उनकी उनके उनको उन्हीं उन्हें उन्होंने उस उसके
उसी उसे एक एवं एस ऐसे और कई कर करता करते करना करने करें कहते कहा का काफ़ी कि
कितना किन्हें किसी किसे की कुछ कुल के को कोई कौन कौनसा गया घर जब जहाँ जा जितना
जिन जिन्हें जिन्होंने जिस जिसे जीधर जैसा जैसे जो तक तब तरह तिन तिन्हें तिन्होंने
तिस तिसे तो था थी थे दबारा दिया दुसरा दूसरे दो द्वारा न नहीं ना निहायत नीचे ने
पर पहले पूरा पे फिर बनी बही बहुत बाद बाला बिलकुल भी भीतर मगर मानो मे में यदि यह
यहाँ यही या यिह ये रखें रहा रहे ऱ्वासा लिए लिये लेकिन व वग़ैरह वर्ग वह वहाँ वहीं
वाले वुह वे वो सकता सकते सबसे सभी साथ साबुत साभ सारा से सो संग है हैं हो होता
होती होते होना होने
""".split())

URL_RE = re.compile(r"http\S+|www\.\S+")
NON_DEVANAGARI_RE = re.compile(r"[^\u0900-\u097F\s]")


def clean_hindi_text(text):
    if not isinstance(text, str):
        return ""
    text = URL_RE.sub(" ", text)
    text = text.replace(chr(0x0964), " ")   # danda ।
    text = text.replace(chr(0x0965), " ")   # double danda ॥
    text = NON_DEVANAGARI_RE.sub(" ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in HINDI_STOPWORDS and len(t) > 1]
    return " ".join(tokens)