import requests
import os
from wav_convertor import convert_to_wav


def test_transcribe_api():
    # URL вашего API
    url = "http://localhost:9000/transcribe"

    # Путь к тестовому аудиофайлу
    audio_file_path = "C:\\Users\\Askar\\VS_CODE\\Python\\whisper_streaming-soundcard\\Demo.mp3"  # Замените на реальный путь к вашему тестовому аудиофайлу
    audio_path = convert_to_wav(audio_file_path)
    
    # Проверяем существование файла
    if not os.path.exists(audio_path):
        print(f"Ошибка: Файл {audio_path} не найден.")
        return

    # Открываем файл в бинарном режиме
    with open(audio_path, "rb") as audio_file:
        # Создаем словарь с файлом для отправки
        files = {"audio_file": (os.path.basename(audio_path), audio_file, "audio/wav")}
        
        # Отправляем POST запрос
        response = requests.post(url, files=files)

    # Проверяем статус ответа
    if response.status_code == 200:
        # Если запрос успешен, выводим результат
        result = response.json()
        print("Тест пройден успешно!")
        print("Транскрибированный текст:", result["text"])
    else:
        # Если произошла ошибка, выводим информацию об ошибке
        print("Тест не пройден.")
        print("Код ошибки:", response.status_code)
        print("Детали ошибки:", response.text)

if __name__ == "__main__":
    test_transcribe_api()