from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
import torch

hdemucs = HDEMUCS_HIGH_MUSDB_PLUS.get_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

