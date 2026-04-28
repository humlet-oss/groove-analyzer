import os
# TensorFlowの互換性設定
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import tempfile
import warnings
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Groove & Dynamics Analyzer", layout="wide")

st.title("🎸 Groove & Dynamics Analyzer (Cloud Edition)")
st.write("WAVファイルをアップロードするだけで、AIが自動解析。演奏のズレと強弱を可視化します。")

# --- UI ---
st.sidebar.header("解析設定")
target_bpm = st.sidebar.number_input("正解のBPM", value=120, min_value=1)
wav_file = st.sidebar.file_uploader("ベースのWAVファイルをアップロード", type=['wav'])

if wav_file is not None:
    with st.spinner("AIが解析中...（1分ほどかかる場合があります）"):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(wav_file.getvalue())
            tmp_path = tmp.name

        try:
            # 1. 音声解析
            y, sr = librosa.load(tmp_path, sr=None)
            tempo_est, _ = librosa.beat.beat_track(y=y, sr=sr)
            estimated_bpm = tempo_est[0] if isinstance(tempo_est, np.ndarray) else tempo_est
            rms = librosa.feature.rms(y=y)[0]
            times_rms = librosa.frames_to_time(range(len(rms)), sr=sr)

            # 2. AIによるMIDI変換
            _, midi_data, _ = predict(tmp_path, ICASSP_2022_MODEL_PATH)
            
            # 3. グルーヴ計算
            notes_data = []
            sixteenth_note_sec = (60.0 / target_bpm) / 4.0
            for note in midi_data.instruments[0].notes:
                onset_time = note.start
                grid_index = round(onset_time / sixteenth_note_sec)
                ideal_time = grid_index * sixteenth_note_sec
                deviation_ms = (onset_time - ideal_time) * 1000
                notes_data.append({"time_sec": onset_time, "deviation_ms": deviation_ms})
            
            df_notes = pd.DataFrame(notes_data)

            # 4. ハイブリッド結合
            dynamics_list = [rms[np.argmin(np.abs(times_rms - t))] for t in df_notes["time_sec"]]
            df_notes["dynamics_score"] = (np.array(dynamics_list) / np.max(dynamics_list)) * 100

            # 5. 結果表示（ダイジェスト）
            st.success("解析完了！")
            bpm_diff = estimated_bpm - target_bpm
            col1, col2, col3 = st.columns(3)
            col1.metric("Target BPM", f"{target_bpm}")
            col2.metric("Estimated BPM", f"{estimated_bpm:.1f}")
            col3.metric("BPM Diff", f"{bpm_diff:+.1f}", delta=f"{bpm_diff:+.1f}", delta_color="inverse")

            st.plotly_chart(px.histogram(df_notes, x="deviation_ms", nbins=30, title="Timing Deviation (ms)"), use_container_width=True)
            st.plotly_chart(px.scatter(df_notes, x="time_sec", y="deviation_ms", size="dynamics_score", color="dynamics_score", title="Groove & Dynamics Timeline"), use_container_width=True)

        except Exception as e:
            st.error(f"エラー: {e}")
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)