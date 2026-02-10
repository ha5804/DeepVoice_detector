import os
import random

def get_data(model="anomaly", root=None, seed=42):
    random.seed(seed)

    speaker_dirs = [
        os.path.join(root, d)
        for d in os.listdir(root)
        if d.isdigit() and os.path.isdir(os.path.join(root, d))
    ]

    random.shuffle(speaker_dirs)

    train_speakers = speaker_dirs[0:100]
    test_speakers  = speaker_dirs[100:200]

    def collect_files(speakers):
        flacs, wavs = [], []
        for spk in speakers:
            for r, _, files in os.walk(spk):
                for f in files:
                    if f.endswith(".flac"):
                        flacs.append(os.path.join(r, f))
        wav_root = os.path.join(root, "wavs")
        if os.path.isdir(wav_root):
            for r, _, files in os.walk(wav_root):
                for f in files:
                    if f.endswith(".wav"):
                        wavs.append(os.path.join(r, f))
        return flacs, wavs

    train_flac, train_wav = collect_files(train_speakers)
    test_flac,  test_wav  = collect_files(test_speakers)

    # ---------------- ANOMALY ----------------
    if model == "anomaly":
        train = train_flac

        test = (
            [(p, 0) for p in test_flac] +
            [(p, 1) for p in test_wav]
        )
        random.shuffle(test)

        return train, test

    # ---------------- CLASSIFIER ----------------
    elif model == "classifier":
        train = (
            [(p, 0) for p in train_flac] +
            [(p, 1) for p in train_wav]
        )
        test = (
            [(p, 0) for p in test_flac] +
            [(p, 1) for p in test_wav]
        )

        random.shuffle(train)
        random.shuffle(test)

        n = len(train)
        n_val = int(0.2 * n)

        return train[:-n_val], train[-n_val:], test

    else:
        raise ValueError("model must be 'anomaly' or 'classifier'")
