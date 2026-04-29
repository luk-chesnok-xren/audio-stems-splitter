import torch
from torchaudio.transforms import Fade

def separate_source(
    model,
    audio,
    segment,
    overlap,
    sample_rate,
    batch_size=16,
    device=None
):  
    chunk_frames = int(sample_rate * (segment + 2*overlap))
    overlap_frames = int(overlap * sample_rate)
    step = int(segment * sample_rate)
    
    batch, channels, length = audio.shape
    
    start = 0
    end = chunk_frames - overlap_frames
    
    fade = Fade(fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape="linear")
    
    result = torch.zeros(batch, len(model.sources), channels, length, device=device)
    
    while end < length:
        chunk = audio[:, :, start:end]
        chunk = chunk.to(device)
        
        with torch.no_grad():           
            out = model.forward(chunk)
        
        out = fade(out)
        
        result[:, :, :, start:end] += out

        if start == 0:
            fade.fade_in_len = int(overlap_frames)
            start += step
        else:  
            start += step + overlap_frames
            
        end = start + chunk_frames
        
        if end >= length-overlap_frames:
            fade.fade_out_len = 0
            
    return result
    
    
    
    
    


