import os
import librosa
from config import LJSPEECH_PATH

# wav 파일 리스트 가져오기
wav_files = os.listdir(LJSPEECH_PATH)
print("총 wav 파일 수:", len(wav_files))

# 첫 번째 파일 선택
wav_path = os.path.join(LJSPEECH_PATH, wav_files[0])
print("불러올 파일:", wav_path)

# 음성 로드
y, sr = librosa.load(wav_path, sr=None)

print("샘플레이트:", sr)
print("신호 길이:", len(y))
print("재생 시간(초):", len(y) / sr)

