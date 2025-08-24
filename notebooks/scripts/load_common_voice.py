# --- Common Voice (Serbian) quick loader + player for Colab ---

from pathlib import Path
import pandas as pd
from IPython.display import Audio, display


class CommonVoiceDataset:
    def __init__(self):
        self.df_ok = None
        self.base_dir = None
        
    def load_dataset(self, path: str):
        """
        Load Common Voice dataset by scanning the clips folder for audio files.
        
        Args:
            path: Path to the extracted Common Voice directory (should contain 'clips/' folder)
        """
        self.base_dir = Path(path)
        clips_dir = self.base_dir / "clips"
        
        if not clips_dir.exists():
            raise FileNotFoundError(f"Clips directory not found: {clips_dir}")
        
        # Find all MP3 files (prioritize) and WAV files as fallback
        audio_files = list(clips_dir.glob("*.mp3")) + list(clips_dir.glob("*.wav"))
        
        if not audio_files:
            raise FileNotFoundError(f"No audio files (.mp3 or .wav) found in {clips_dir}")
        
        print(f"Found {len(audio_files)} audio files")
        
        # Create simple dataframe with audio paths
        self.df_ok = pd.DataFrame({
            'audio_path': audio_files,
            'filename': [f.name for f in audio_files]
        })
        
        return self.df_ok

    def play(self, idx: int, rate: int | None = None):
        """
        Play sample at df_ok index `idx`.
        Optionally specify `rate` to set playback sample rate for the widget (None = auto).
        """
        if self.df_ok is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
            
        row = self.df_ok.iloc[idx]
        print(f"[{idx}] {row['filename']}")
        display(Audio(filename=str(row["audio_path"]), rate=rate, autoplay=False))
