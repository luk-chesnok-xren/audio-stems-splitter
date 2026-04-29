from flask import Flask, request, jsonify, send_file
import torch
import torchaudio
import io
import zipfile
import librosa
import soundfile as sf

from model.chunking import separate_source
from model.hdemucs import hdemucs, device
from model.config import *

app = Flask(__name__)

@app.route('/separate', methods=['POST'])
def separate():
    if 'file' not in request.files:
        return jsonify({'Ошибка': 'файл отсутствует'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'Ошибка': 'название файла не должно быть пустым и должно содержать расширение (.wav/mp3)'}), 400
    
    allowed = {'mp3', 'wav'}
    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in allowed:
        return jsonify({'Ошибка': f'неподдерживаемый формат файла: {ext}'}), 400
    
    try:
        audio, sample_rate = librosa.load(io.BytesIO(file.read()), sr=None, mono=False)
        
        if (audio.shape[-1] / sample_rate > MAX_DURATION) or audio.shape[-1] / sample_rate < MIN_DURATION:
            return jsonify({'Ошибка': 'аудиофайл не должен иметь длительность менее трёх минут либо более десяти минут.'})
    except Exception:
        return jsonify({'error': 'не удалось прочесть загруженный файл'}), 400
    
    audio, sample_rate = librosa.load(io.BytesIO(file.read()), sr=None, mono=False)
    
    waveform = torch.tensor(audio).unsqueeze(0)
    
    #нормализация данных сэмпла
    ref = waveform.mean(0)
    waveform = (waveform - ref.mean()) / ref.std()
    waveform = waveform.to(device)
    
    sources = separate_source(hdemucs, waveform, SEGMENT, OVERLAP, sample_rate, device=device).squeeze(0)
    
    #денормализация итоговых сэмплов и приведение к исходному масштабу
    ref = ref.to(device)
    sources = sources * ref.std() + ref.mean()
    sources = sources.cpu()
    
    source_names = ["drums", "bass", "other", "vocals"]
    
    #формирование zip архива 
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zf:
        for name, source in zip(source_names, sources):
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, source.T.numpy(), sample_rate, format='wav')
            zf.writestr(f'{name}.wav', wav_buffer.getvalue())
    
    zip_buffer.seek(0)
    #ответ веб-сервера
    return send_file(zip_buffer, mimetype='application/zip', download_name='stems.zip')

if __name__ == '__main__':
    app.run()