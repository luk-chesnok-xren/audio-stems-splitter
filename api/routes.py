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
        return jsonify({'error': 'no file'}), 400
    
    file = request.files['file']
    
    audio, sample_rate = librosa.load(io.BytesIO(file.read()), sr=44100, mono=False)
    
    waveform = torch.tensor(audio).unsqueeze(0)
    
    ref = waveform.mean(0)
    waveform = (waveform - ref.mean()) / ref.std()
    waveform = waveform.to(device)
    
    sources = separate_source(hdemucs, waveform, SEGMENT, OVERLAP, sample_rate, device=device).squeeze(0)
    
    ref = ref.to(device)
    sources = sources * ref.std() + ref.mean()
    sources = sources.cpu()
    
    source_names = ["drums", "bass", "other", "vocals"]
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zf:
        for name, source in zip(source_names, sources):
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, source.T.numpy(), sample_rate, format='wav')
            zf.writestr(f'{name}.wav', wav_buffer.getvalue())
    
    zip_buffer.seek(0)
    return send_file(zip_buffer, mimetype='application/zip', download_name='stems.zip')

if __name__ == '__main__':
    app.run()