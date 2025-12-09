import streamlit as st
import pandas as pd
import subprocess
import os
import time
from datetime import datetime
import pypianoroll
import matplotlib.pyplot as plt
from midi2audio import FluidSynth
from miditoolkit import MidiFile
from pydub import AudioSegment  # 追加
from pydub.effects import normalize  # 追加

# ==========================================
# 設定・定数
# ==========================================
# 実験データの保存先
DATA_LOG_FILE = "experiment1_results.csv"

# サウンドフォントのパス (プロジェクト内のパスに合わせてください)
SOUND_FONT_PATH = "EMO-Disentanger/SalamanderGrandPiano-SF2-V3+20200602/SalamanderGrandPiano-V3+20200602.sf2"

# 一時ファイル保存用ディレクトリ
TEMP_DIR = "experiment_temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# ページ設定
st.set_page_config(
    page_title="EMO-Music 評価実験",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 関数定義
# ==========================================

def convert_midi_to_wav(midi_path, wav_path):
    """
    MIDIをWAVに変換し、音量をノーマライズ（最大化）する。
    1. Channel 9 などの音色不整合を修正
    2. FluidSynth で WAV 化
    3. pydub で音量をブースト (Normalize)
    """
    try:
        # --- Step 1: MIDIファイルの修正 (音色割り当て) ---
        midi_obj = MidiFile(midi_path)
        
        for instrument in midi_obj.instruments:
            # 強制的にピアノ(Program 0)に設定し、ドラムフラグを折る
            instrument.program = 0
            instrument.is_drum = False
        
        # 修正したMIDIを一時保存
        fixed_midi_path = midi_path.replace(".mid", "_fixed.mid")
        midi_obj.dump(fixed_midi_path)
        
        # --- Step 2: WAV変換 (FluidSynth) ---
        fs = FluidSynth(SOUND_FONT_PATH)
        fs.midi_to_audio(fixed_midi_path, wav_path)
        
        # --- Step 3: 音量ノーマライズ (pydub) ---
        # 生成されたWAVを読み込む
        audio = AudioSegment.from_wav(wav_path)
        
        # ノーマライズ（ピークを 0dBFS に合わせる＝音割れしない最大音量にする）
        normalized_audio = normalize(audio)
        
        # さらに少しブーストしたい場合は以下のようにゲインを追加（お好みで）
        # normalized_audio = normalized_audio + 3  # +3dB
        
        # 上書き保存
        normalized_audio.export(wav_path, format="wav")
        
        return True
    except Exception as e:
        st.error(f"WAV変換/音量調整エラー: {e}")
        return False

def visualize_pianoroll(midi_path):
    """MIDIをピアノロールとして表示する"""
    try:
        multitrack = pypianoroll.read(midi_path)
        if len(multitrack.tracks) > 0:
            fig, ax = plt.subplots(figsize=(10, 3))
            pypianoroll.plot_track(multitrack.tracks, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("MIDIトラックが空です")
    except Exception as e:
        st.error(f"可視化エラー: {e}")

def run_generation(midi_file_path, emotion_label):
    """run_pipeline.sh を実行する"""
    cmd = ["bash", "run_pipeline.sh", midi_file_path, emotion_label]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode!= 0:
            st.error("生成エラーが発生しました")
            st.code(result.stderr)
            return None
        
        basename = os.path.splitext(os.path.basename(midi_file_path))
        output_dir = "EMO-Disentanger/generation/pipeline_output"
        
        # ファイル名パターン構築
        target_file_name = f"harm_{basename}_{emotion_label}_full.mid"
        target_path = os.path.join(output_dir, target_file_name)
        
        if os.path.exists(target_path):
            return target_path
        else:
            st.warning(f"指定ファイル {target_file_name} が見つかりません。最新の生成物を探索します。")
            if not os.path.exists(output_dir):
                 return None
            files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".mid")]
            if not files:
                return None
            latest_file = max(files, key=os.path.getctime)
            return latest_file

    except Exception as e:
        st.error(f"実行時例外: {e}")
        return None

def save_evaluation(user_id, input_filename, emotion, ratings, feedback):
    """評価データをCSVに追記保存"""
    data = {
        "timestamp": [datetime.now()],
        "user_id": [user_id],
        "input_midi": [input_filename],
        "emotion_condition": [emotion],
        "score_emotion_match": [ratings["emotion"]],
        "score_consistency": [ratings["consistency"]],
        "score_control": [ratings["control"]],
        "score_usefulness": [ratings["usefulness"]],
        "feedback": [feedback]
    }
    df = pd.DataFrame(data)
    
    if not os.path.isfile(DATA_LOG_FILE):
        df.to_csv(DATA_LOG_FILE, index=False)
    else:
        df.to_csv(DATA_LOG_FILE, mode='a', header=False, index=False)

# ==========================================
# メイン UI
# ==========================================

st.title("🎹 EMO-Music 生成システム評価実験")
st.markdown("""
この実験では、あなたが入力したメロディに対して、AIが特定の**感情 (Emotion)** に合わせた伴奏を生成します。
生成された楽曲を聴き、その品質やシステムの有用性を評価してください。
""")

# サイドバー: ユーザー情報
with st.sidebar:
    st.header("実験者情報")
    user_id = st.text_input("ユーザーID (氏名または識別子)", value="guest")
    st.info("※IDを入力しないと結果を保存できません")

# --- Step 1: メロディ入力 ---
st.header("Step 1: メロディの入力")
st.markdown("8小節程度の短いメロディ（MIDIファイル）をアップロードしてください。")

uploaded_file = st.file_uploader("MIDIファイルをドラッグ＆ドロップ", type=["mid", "midi"])

if uploaded_file is not None:
    # 一時保存
    temp_input_path = os.path.join(TEMP_DIR, uploaded_file.name)
    with open(temp_input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"読み込み完了: {uploaded_file.name}")
    
    # ピアノロール可視化
    with st.expander("入力メロディの確認 (ピアノロール)", expanded=True):
        visualize_pianoroll(temp_input_path)

    # --- Step 2: 感情選択 & 生成 ---
    st.header("Step 2: 感情の指定と生成")
    
    emotion_options = {
        "Q1": "Q1: 喜び (Joy) - 明るくエネルギッシュ",
        "Q2": "Q2: 怒り (Anger/Tension) - 激しく緊張感がある",
        "Q3": "Q3: 悲しみ (Sadness) - 暗く静か",
        "Q4": "Q4: 楽しい (Happy/Relax) - 明るく落ち着いている"
    }
    
    selected_emotion_key = st.radio(
        "生成したい感情を選んでください",
        list(emotion_options.keys()),
        format_func=lambda x: emotion_options[x]
    )

    if st.button("🚀 伴奏を生成する", type="primary"):
        if not user_id:
            st.error("サイドバーでユーザーIDを入力してください。")
        else:
            with st.status("AIが作曲中...", expanded=True) as status:
                st.write("1. 和声進行を生成中 (EMO_Harmonizer)...")
                # パイプライン実行
                midi_output_path = run_generation(temp_input_path, selected_emotion_key)
                
                if midi_output_path:
                    st.write("2. 伴奏を生成中 (Stage 2)...")
                    st.write("3. 音声ファイルに変換・音量調整中 (Normalization)...")
                    
                    # WAV変換 (修正済みの関数を使用)
                    wav_output_path = midi_output_path.replace(".mid", ".wav")
                    success = convert_midi_to_wav(midi_output_path, wav_output_path)
                    
                    if success:
                        status.update(label="生成完了！", state="complete", expanded=False)
                        
                        # セッションステートに保存して再描画対策
                        st.session_state['generated_wav'] = wav_output_path
                        st.session_state['current_emotion'] = selected_emotion_key
                        st.session_state['current_midi_name'] = uploaded_file.name
                    else:
                        status.update(label="WAV変換に失敗しました", state="error")
                else:
                    status.update(label="生成に失敗しました", state="error")

    # --- Step 3: 評価フォーム ---
    if 'generated_wav' in st.session_state:
        st.divider()
        st.header("Step 3: 試聴と評価")
        
        st.markdown(f"**生成された条件:** `{st.session_state['current_emotion']}` (入力: `{st.session_state['current_midi_name']}`)")
        
        # プレイヤー
        st.audio(st.session_state['generated_wav'], format="audio/wav")
        
        # 評価フォーム
        with st.form("eval_form"):
            st.subheader("アンケート")
            
            c1, c2 = st.columns(2)
            with c1:
                score_emotion = st.slider(
                    "1. 感情一致度: 指定した感情に合っていますか？",
                    1, 5, 3, help="1: 全く合っていない 〜 5: 非常に合っている"
                )
                score_consistency = st.slider(
                    "2. 一貫性: メロディと伴奏は馴染んでいますか？",
                    1, 5, 3, help="1: 違和感がある 〜 5: 自然である"
                )
            with c2:
                score_control = st.slider(
                    "3. 操作感: 意図通りにコントロールできたと感じますか？",
                    1, 5, 3
                )
                score_usefulness = st.slider(
                    "4. 有用性: 作曲支援ツールとして役に立ちそうですか？",
                    1, 5, 3
                )
            
            feedback = st.text_area("自由記述 (気になった点、改善点など)")
            
            submitted = st.form_submit_button("評価を送信してリセット")
            
            if submitted:
                save_evaluation(
                    user_id,
                    st.session_state['current_midi_name'],
                    st.session_state['current_emotion'],
                    {
                        "emotion": score_emotion,
                        "consistency": score_consistency,
                        "control": score_control,
                        "usefulness": score_usefulness
                    },
                    feedback
                )
                st.success("評価を保存しました！次の条件を試してください。")
                # ステートをクリアしてリセット
                del st.session_state['generated_wav']
                time.sleep(1)
                st.rerun()
