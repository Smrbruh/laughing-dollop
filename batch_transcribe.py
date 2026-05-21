import os
import sys
import yt_dlp
from faster_whisper import WhisperModel

def get_video_id(url):
    return url.split("v=")[-1].split("&")[0]

def transcribe_video(url, model, output_dir):
    video_id = get_video_id(url)
    txt_file = os.path.join(output_dir, f"{video_id}.txt")

    if os.path.exists(txt_file):
        print(f"  Уже готово: {txt_file}")
        return

    print(f"  Скачиваю аудио...")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f"{video_id}.%(ext)s",
        'quiet': True,
        'no_warnings': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
        }],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        print(f"  Ошибка скачивания: {e}")
        return

    actual_file = None
    for ext in ['mp3', 'webm', 'm4a', 'opus']:
        candidate = f"{video_id}.{ext}"
        if os.path.exists(candidate):
            actual_file = candidate
            break

    if not actual_file:
        print("  Аудиофайл не найден!")
        return

    print(f"  Расшифровываю (en)...")
    segments, info = model.transcribe(actual_file, language="en", beam_size=5)

    with open(txt_file, 'w', encoding='utf-8') as f:
        for segment in segments:
            f.write(f"{segment.text}\n")

    os.remove(actual_file)
    print(f"  Готово: {txt_file}")


def main():
    links_file = "links.txt"
    output_dir = "transcripts"

    if not os.path.exists(links_file):
        print(f"Файл {links_file} не найден!")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    with open(links_file, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]

    print("Загружаю модель faster-whisper (small)...")
    model = WhisperModel("small", device="cpu", compute_type="int8")

    total = len(urls)
    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{total}] {url}")
        transcribe_video(url, model, output_dir)

    print(f"\nВсё готово! Файлы в папке: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()