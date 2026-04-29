import torch
from torchaudio.transforms import Fade
import torch.nn.functional as F

def separate_source(
    model,
    audio,
    segment,
    overlap,
    sample_rate,
    batch_size=8,
    device=None
):  
    """
    def separate_source(
        model, - модель для разделения сэмпла на стемы
        audio, - данные аудиофайла в формате pytorch тензора
        segment, - размер сегмента в секундах (чистый сигнал без изменения уровня сигнала)
        overlap, - длительность линейного изменения уровня сигнала с края сегмента в секундах
        sample_rate, - частота дискретизации сэмпла
        batch_size=8,-раазмер батча, прогоняемого через модель
        device=None - вычислительное устройство
    )
    
    функция реализует батчинг одного трека и его ввод в модель сегментами
    
    данная функция возвращает тензор с 4 стемами в формате 
    [B, sources, channels, length] 
    """
    chunk_frames = int(sample_rate * (segment + 2*overlap))
    overlap_frames = int(overlap * sample_rate)
    step = int(segment * sample_rate)
    
    batch, channels, length = audio.shape
    
    def batched_forward_audio():
        chunks = []
        positions = []
        
        start = 0
        end = chunk_frames - overlap_frames
        
        while end < length:
            #формирование сегментов и сохранение начальных сегментов
            chunk = audio[:, :, start:end]
            chunks.append(chunk)
            positions.append(start)

            #"накладка" конца каждого сегмента на начало следующего сегмента
            #реализует cross fade на стыках сегментах
            #так как модель склонна искажать сигнал на краях сегмента 
            if start == 0:
                start += step
            else:
                start += step + overlap_frames

            end = start + chunk_frames
        
        #получение вывода для каждого батча сегментов
        for i in range(0, len(chunks), batch_size):
            batch_list = chunks[i:i+batch_size]
            
            #паддинг для последнего сегмента если он меньше стандартного сегмента с учетом накладок
            batch_list = [F.pad(c, (0, chunk_frames - c.shape[-1])) if c.shape[-1] < chunk_frames else c for c in batch_list]
            
            batch = torch.cat(batch_list, dim=0).to(device)
            with torch.no_grad():
                batch_out = model.forward(batch)
                  
            yield batch_out, positions[i:i+batch_size]
            
    result = torch.zeros(batch, len(model.sources), channels, length, device=device)
    fade = Fade(fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape="linear")
    
    #формирование результата 
    for batch_out, positions in batched_forward_audio():
        for i, (chunk, pos) in enumerate(zip(batch_out, positions)):
            if i != 0: 
                fade.fade_in_len = overlap_frames
            elif i == len(positions) - 1:
                fade.fade_out_len = 0
            
            result[:, :, :, pos:pos+batch_out.shape[-1]] += fade(chunk)
            
    return result
    
    
    
    
    


