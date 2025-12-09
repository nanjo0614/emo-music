import streamlit as st
import pandas as pd
import subprocess
import os
import time
import glob
from datetime import datetime
import matplotlib.pyplot as plt
from midi2audio import FluidSynth
from miditoolkit import MidiFile
from pydub import AudioSegment
from pydub.effects import normalize

# ==========================================
# 🔧 設定・定数
# ==========================================

# 評価結果の保存先CSV
DATA_LOG_FILE = "experiment1_evaluation_log.csv"

# サウンドフォントのパス (環境に合わせてパスを確認してください)
SOUND_FONT_PATH = "EMO-Disentanger/SalamanderGrandPiano-SF2-V3+20200602/SalamanderGrandPiano-V3+20200602.sf2"

# 感情ラベル定義
EMOTIONS = ["Q1", "Q2", "Q3", "Q4"]
EMOTION_DISPLAY = {
    "Q1": "Q1: 喜び (Joy)",
    "Q2": "Q2: 怒り (Anger)",
    "Q3": "Q3: 悲しみ (Sadness)",
    "Q4": "Q4: 楽しい (Fun)"
}

# 一時フォルダ
TEMP_DIR = "experiment_temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# ページ設定
st.set_page_config(
    page_title="EMO-Music 評価実験システム",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 🛠 内部処理関数
# ==========================================

def cleanup_output_directories():
    """
    データの混入を防ぐため、生成前に過去の出力ファイルを削除する
    """
    target_dirs = [
        "EMO-Disentanger/generation/pipeline_output",
        "EMO_Harmonizer/generation/pipeline_temp"
    ]
    
    deleted_count = 0
    for d in target_dirs:
        if os.path.exists(d):
            files = glob.glob(os.path.join(d, "*"))
            for f in files:
                try:
                    os.remove(f)
                    deleted_count += 1
                except Exception as e:
                    print(f"削除エラー: {e}")
    
    print(f"[Info] Cleaned up {deleted_count} old files.")

def convert_midi_to_wav(midi_path, wav_path):
    """MIDIをWAVに変換し、ノーマライズする"""
    try:
        # 1. MIDI修正 (Program Change等)
        midi_obj = MidiFile(midi_path)
        for instrument in midi_obj.instruments:
            instrument.program = 0
            instrument.is_drum = False
        fixed_midi_path = midi_path.replace(".mid", "_fixed.mid")
        midi_obj.dump(fixed_midi_path)
        
        # 2. FluidSynthでWAV化
        fs = FluidSynth(SOUND_FONT_PATH)
        fs.midi_to_audio(fixed_midi_path, wav_path)
        
        # 3. Pydubでノーマライズ
        audio = AudioSegment.from_wav(wav_path)
        normalized_audio = normalize(audio)
        normalized_audio.export(wav_path, format="wav")
        return True
    except Exception as e:
        print(f"[Error] WAV Conversion failed: {e}")
        return False

def run_generation_pipeline(input_midi_path, emotion):
    """
    run_pipeline.shを実行し、生成されたMIDIファイルのパスを返す。
    ディレクトリ内の最新ファイルを拾う救済ロジック付き。
    """
    cmd = ["bash", "run_pipeline.sh", input_midi_path, emotion]
    
    try:
        # パイプライン実行
        subprocess.run(cmd, capture_output=True, text=True)
        
        # ファイル探索 (EMO-Disentanger内の出力先)
        output_dir = "EMO-Disentanger/generation/pipeline_output"
        
        # 救済策: 厳密なファイル名指定ではなく、フォルダ内の更新日時が新しいものを拾う
        if os.path.exists(output_dir):
            files = [
                os.path.join(output_dir, f) 
                for f in os.listdir(output_dir) 
                if f.endswith(".mid") and emotion in f
            ]
            if files:
                # 最も新しいファイルを返す
                return max(files, key=os.path.getctime)
        
        return None
    except Exception as e:
        st.error(f"System Error: {e}")
        return None

def save_to_csv(data_dict):
    """評価データをCSVに追記"""
    df = pd.DataFrame([data_dict])
    if not os.path.isfile(DATA_LOG_FILE):
        df.to_csv(DATA_LOG_FILE, index=False)
    else:
        df.to_csv(DATA_LOG_FILE, mode='a', header=False, index=False)

def display_player_and_download(emotion_key):
    """
    プレイヤーとダウンロードボタンを表示する (フォームの外で使用)
    """
    st.markdown(f"### {EMOTION_DISPLAY[emotion_key]}")
    res = st.session_state.results.get(emotion_key)
    
    if res:
        # プレイヤー
        st.audio(res['wav'], format="audio/wav")
        
        # ダウンロードボタン
        fixed_midi_path = res['midi'].replace(".mid", "_fixed.mid")
        if os.path.exists(fixed_midi_path):
            with open(fixed_midi_path, "rb") as f:
                st.download_button(
                    label=f"📥 MIDI DL ({emotion_key})",
                    data=f,
                    file_name=os.path.basename(fixed_midi_path),
                    mime="audio/midi",
                    key=f"dl_{emotion_key}"
                )
    else:
        st.error("生成失敗")
    st.divider()

def render_slider(emotion_key, ratings_dict):
    """
    スライダーのみを表示する (フォームの中で使用)
    """
    st.caption(f"{EMOTION_DISPLAY[emotion_key]} の評価")
    
    if st.session_state.results.get(emotion_key):
        ratings_dict[emotion_key] = st.slider(
            f"「{emotion_key}」らしさを感じますか？",
            1, 5, 3, 
            key=f"q_emo_{emotion_key}",
            help="1:全く感じない 〜 5:強く感じる"
        )
    else:
        ratings_dict[emotion_key] = None
    st.markdown("---")


# ==========================================
# 🖥 UI構築 (メイン部)
# ==========================================

st.title("🎹 EMO-Music(仮) 評価実験")
st.markdown("""
ご協力ありがとうございます。この実験では、あなたが入力したメロディに対して、システムが感情に応じた伴奏を生成します。
以下の手順に従って評価をお願いします。
""")

# --- サイドバー: ユーザー登録 ---
with st.sidebar:
    st.header("1. 実験者登録")
    user_id = st.text_input("ユーザーID (氏名)", key="user_id")
    
    st.divider()
    st.info("""
    **実験の流れ:**
    1. 4~8小節程度のメロディを入力
    2. 「生成開始」ボタンを押す
    3. 4種類の伴奏を聴き比べる
    4. アンケートに回答する
    """)

if not user_id:
    st.warning("👈 左のサイドバーで「ユーザーID」を入力してください。")
    st.stop()

# --- メインエリア: タスク進行 ---

if "experiment_phase" not in st.session_state:
    st.session_state.experiment_phase = "input"
if "results" not in st.session_state:
    st.session_state.results = {}

# === Phase 1: ファイル入力 ===
st.header("Step 1: メロディの入力")
uploaded_file = st.file_uploader("4~8小節程度のメロディ(MIDI)をアップロードしてください", type=["mid", "midi"])

if uploaded_file:
    # ファイル保存
    temp_path = os.path.join(TEMP_DIR, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.audio(uploaded_file, format="audio/midi")

    # 生成ボタン
    if st.button("🚀 喜怒哀楽の4パターンの伴奏を一括生成する (数分かかります)", type="primary"):
        # ★ここで前回データを削除
        cleanup_output_directories()
        
        st.session_state.experiment_phase = "generating"
        st.rerun()

# === Phase 2: 生成処理 ===
if st.session_state.experiment_phase == "generating":
    st.markdown("---")
    st.header("⏳ 生成中...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = {}
    temp_path = os.path.join(TEMP_DIR, uploaded_file.name)
    
    for i, emo in enumerate(EMOTIONS):
        status_text.text(f"生成中: {EMOTION_DISPLAY[emo]} ({i+1}/4)")
        midi_out = run_generation_pipeline(temp_path, emo)
        
        if midi_out:
            wav_out = midi_out.replace(".mid", ".wav")
            convert_midi_to_wav(midi_out, wav_out)
            results[emo] = {"midi": midi_out, "wav": wav_out}
        else:
            results[emo] = None
        
        progress_bar.progress((i + 1) / 4)
    
    st.session_state.results = results
    st.session_state.experiment_phase = "evaluation"
    st.rerun()

# === Phase 3: 聴取と評価 (アンケート) ===
if st.session_state.experiment_phase == "evaluation":
    st.markdown("---")
    st.header("Step 2: 聴き比べと評価")
    st.markdown("生成された4つの楽曲を聴き、以下のアンケートに答えてください。")

    # ---------------------------------------------------------
    # 1. プレイヤーとDLボタンの表示 (フォームの外)
    # ---------------------------------------------------------
    st.subheader("1. 試聴")
    
    # 上段: Q2(怒り), Q1(喜び)
    c_ul, c_ur = st.columns(2)
    with c_ul: display_player_and_download("Q2")
    with c_ur: display_player_and_download("Q1")
    
    # 下段: Q3(悲しみ), Q4(楽しさ)
    c_ll, c_lr = st.columns(2)
    with c_ll: display_player_and_download("Q3")
    with c_lr: display_player_and_download("Q4")

    # ---------------------------------------------------------
    # 2. 評価フォーム (ここから st.form)
    # ---------------------------------------------------------
    st.subheader("2. 評価アンケート")
    
    with st.form("evaluation_form"):
        st.markdown("それぞれの楽曲について、指定された**感情ラベル**に合っているか評価してください。")
        
        # 感情一致度評価
        ratings_emotion = {}
        
        col1, col2 = st.columns(2)
        with col1:
            render_slider("Q2", ratings_emotion)
            render_slider("Q3", ratings_emotion)
        with col2:
            render_slider("Q1", ratings_emotion)
            render_slider("Q4", ratings_emotion)

        # 全体評価
        st.markdown("#### システム全体の評価")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**メロディとの一貫性**")
            q_consistency = st.slider(
                "伴奏は入力したメロディに自然に馴染んでいましたか？",
                1, 5, 3
            )
            st.markdown("**操作感 (コントロール性)**")
            q_control = st.slider(
                "感情ラベル(Q1〜Q4)による曲調の変化は明確でしたか？",
                1, 5, 3
            )
        with c2:
            st.markdown("**有用性**")
            q_usefulness = st.slider(
                "作曲支援ツールとして役に立つと思いますか？",
                1, 5, 3
            )
            q_free_text = st.text_area("自由記述 (気になった点、改善点など)")

        # 送信ボタン
        st.markdown("---")
        submitted = st.form_submit_button("評価を送信して終了 (データ保存)", type="primary")
        
        if submitted:
            eval_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user_id": user_id,
                "input_file": uploaded_file.name,
                "score_Q1_match": ratings_emotion.get("Q1"),
                "score_Q2_match": ratings_emotion.get("Q2"),
                "score_Q3_match": ratings_emotion.get("Q3"),
                "score_Q4_match": ratings_emotion.get("Q4"),
                "score_consistency": q_consistency,
                "score_control": q_control,
                "score_usefulness": q_usefulness,
                "feedback": q_free_text
            }
            
            save_to_csv(eval_data)
            
            st.success("✅ 評価データが保存されました。ご協力ありがとうございました！")
            st.balloons()
            
    # リセットボタン (フォームの外)
    if st.button("別の曲で実験を続ける"):
        st.session_state.experiment_phase = "input"
        st.session_state.results = {}
        st.rerun()

# デバッグ用
st.sidebar.markdown("---")
if st.sidebar.button("強制リセット"):
    st.session_state.experiment_phase = "input"
    st.session_state.results = {}
    st.rerun()