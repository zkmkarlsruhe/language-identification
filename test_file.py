"""
:author:
Paul Bethge (bethge@zkm.de)
2021

:License:
This package is published under GNU GPL Version 3.
"""

import scipy.io.wavfile as wav

from src.utils.language_analyzer import LanguageAnalyzer

filename = "test/wav/test_german.wav"
model_path = "trained_models/weights.18"
config_path = "trained_models/config.yaml"

fs, wav = wav.read(filename)
print(wav)

analyzer = LanguageAnalyzer(model_path, config_path)
out = analyzer.predict_on_audio_file(filename)
print(out)