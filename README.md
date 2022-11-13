# <a href="https://githubtocolab.com/MTG/tape/blob/main/colab_demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a> Timbre-Aware Pitch Estimator


TAPE is a novel pitch estimator (or pitch stream segregator) that improves **blind** pitch estimation with the help of timbre-specific priors. As a proof-of-concept, we present it with a real-world use case on violin--piano duets, one of the most common scenarios in chamber music. 

Usage:
```
from tape.pitch_estimator import TAPE
import torch
import librosa
device = 'cuda' if torch.cuda.is_available() else 'cpu'

violin_tape = TAPE(instrument='violin', 
                   window_size=16*1024,   # 1.024 seconds,
                   hop_length=128         # 128/16000= 8 miliseconds
                   ).to(device)

audio, _ = librosa.load('violin-piano.wav', sr=violin_tape.sr)
with torch.no_grad():
  time, frequency, confidence, activation = violin_tape.predict(torch.tensor(audio), 
                                                                viterbi=False, batch_size=128)
```

For real-world violin--piano examples from YouTube, please visit the colab demo. You can also use your own YouTube links to directly sonify the violin pitch track from any recording.

<a href="https://githubtocolab.com/MTG/tape/blob/main/colab_demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>
