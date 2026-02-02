import parselmouth
from config import LJSPEECH_PATH
from parselmouth.praat import call
import os

wav_files = os.list.dir(LJSPEECH_PATH)

sound = parselmouth.Sound(wav_files)

pitch = sound.to_pitch()
mean_pitch = call(pitch, "Get mean", 0, 0, "Hertz")
