# config.py
import os
#파일 경로를 운영체제 독립적으로 다루기 위해 사용
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#config.py가 있는 위치

DATASET_ROOT = DATASET_ROOT = "/Users/taehayeong/Desktop/dataset"
#dataset과 github에 올라갈 파일을 분리하기 위한 설계
#gitgub에는 큰 용량 파일 업로드 제한 존재.

LJSPEECH_PATH = os.path.join(
    DATASET_ROOT, "wavs"
)
#실제 wav파일이 존재하는 폴더. 추후에 LJSPPECH_PATH 변수만 이용