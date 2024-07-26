import os
from pydub import AudioSegment

def convert_to_wav(input_file):
    try:
        # Получаем путь и имя файла без расширения
        file_path, file_name = os.path.split(input_file)
        file_name_without_ext = os.path.splitext(file_name)[0]
        
        # Формируем имя выходного файла
        output_file = os.path.join(file_path, f"{file_name_without_ext}.wav")
        
        # Загружаем аудио файл
        audio = AudioSegment.from_file(input_file)
        
        # Экспортируем в WAV формат
        audio.export(output_file, format="wav")
        
        print(f"Конвертация завершена. Результат сохранен в {output_file}")
        return output_file
    except Exception as e:
        print(f"Произошла ошибка при конвертации: {str(e)}")
        return None
