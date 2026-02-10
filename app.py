import streamlit as st
import torch
import numpy as np

from src.models import LSTMAutoEncoder
from src.utils import extract_logmel
from src.preprocess import AudioPreprocessor  


def split_audio(wav, segment_len):
    n = len(wav) // segment_len
    return [wav[i*segment_len:(i+1)*segment_len] for i in range(n)]


@st.cache_resource
def load_model():
    model = LSTMAutoEncoder(
        n_mels=80,
        hidden_dim=128,
        latent_dim=64,
        num_layers=2
    )
    model.load_state_dict(
        torch.load("checkpoints/lstm_ae.pth", map_location="cpu")
    )
    model.eval()
    return model


model = load_model()

st.title("Deep Voice Protector")

uploaded = st.file_uploader("통화 음성 파일 업로드 (.wav)", type=["wav", "flac"])

if uploaded and st.button("탐지 시작"):
    prep = AudioPreprocessor(sr=16000)
    wav = prep.preprocess(uploaded)

    segments = split_audio(
        wav,
        segment_len=300 * 256
    )

    errors = []

    with st.spinner("AI가 통화 초반 음성을 분석 중입니다..."):
        for seg in segments[:5]:
            feat = torch.tensor(
                extract_logmel(seg, sr=16000, n_mels = 80),
                dtype=torch.float32
            ).unsqueeze(0)

            with torch.no_grad():
                recon = model(feat)
                err = torch.mean((recon - feat) ** 2).item()
                errors.append(err)

    score = float(np.mean(errors))
    risk = min(score / 0.7, 1.0) * 100

    st.subheader("탐지 결과")
    st.write(f"딥보이스 의심 확률: **{risk:.1f}%**")

    if risk >= 70:
        st.error("딥보이스 위험 높음\n\n송금 보류 및 재확인 권장")
    else:
        st.success("정상 음성으로 판단됨")



