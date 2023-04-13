import googletrans
from googletrans import Translator

def translate_to_english(text):
    translator = Translator()
    translated = translator.translate(text)
    return translated.text

# sentence = "एक स्ट्रिंग को उलटने के लिए एक प्रोग्राम लिखें।"
# english_sentence = translate_to_english(sentence)
# print(english_sentence)
# translation, attention = translate_sentence(english_sentence, SRC, TRG, model, device)
# display_attention(sentence, translation, attention)