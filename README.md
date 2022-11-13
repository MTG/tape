# Timbre-Aware Pitch Estimator
<a href="https://githubtocolab.com/MTG/tape/blob/main/colab_demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>


TAPE is a novel pitch estimator (or pitch stream segregator) improving **blind** pitch estimation with the help of timbre-specific priors. As a proof-of-concept, we present it with a real-world use case on violin--piano duets, one of the most common scenarios in chamber music. 

Usage:
```
from tape import pitch_estimator

violin_tape = pitch_estimator.TAPE(instrument='violin')
```

More examples can be found in the colab demo.
