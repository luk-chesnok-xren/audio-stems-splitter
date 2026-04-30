# 🎵 Audio Stem Splitter

REST API для разделения аудио на 4 стема (vocals, drums, bass, other) на основе модели [HDemucs](https://github.com/facebookresearch/demucs).

## Быстрый старт

### Требования

- Python 3.10+
- [uv](https://github.com/astral-sh/uv)

### Установка

```bash
git clone https://github.com/yourname/audio-stem-splitter
cd audio-stem-splitter
uv sync
```

### Запуск

```bash
uv run -m api.routes
```

Сервер запустится на `http://localhost:5000`.

## API

### `POST /separate`

Разделяет аудиофайл на 4 стема.

**Request:** `multipart/form-data`

| Поле | Тип | Описание |
|------|-----|----------|
| `file` | file | Аудиофайл (mp3, wav, flac, ogg) |

**Response:** `application/zip` — архив `stems.zip` с файлами:
- `vocals.wav`
- `drums.wav`
- `bass.wav`
- `other.wav`

### Пример

```bash
curl -X POST http://localhost:5000/separate \
  -F "file=@track.mp3" \
  --output stems.zip
```

### Ограничения

- Максимальная длина трека: 10 минут
- Поддерживаемые форматы: mp3, wav, flac, ogg

## Метрики

Оценка на 10 треках из [MUSDB18](https://zenodo.org/record/1117372) (медиана по фреймам, медиана по трекам):

| Stem   | SDR   | SIR    | ISR    | SAR   |
|--------|-------|--------|--------|-------|
| vocals | 8.123 | 13.734 | 9.308  | 5.942 |
| drums  | 9.017 | 12.873 | 11.022 | 6.664 |
| bass   | 9.733 | 17.024 | 10.384 | 7.037 |
| other  | 5.701 | 9.852  | 7.022  | 4.367 |

- **SDR** (Signal-to-Distortion Ratio) — общее качество разделения
- **SIR** (Signal-to-Interference Ratio) — утечка других инструментов
- **ISR** (Image-to-Spatial Ratio) — сохранение стерео образа
- **SAR** (Sources-to-Artifacts Ratio) — артефакты модели

## Производительность

| Железо | Время на 4 мин трек |
|--------|---------------------|
| Intel Core i5 1335U (CPU only) | ~45 сек |