#!/bin/bash

# ==========================================
# EMO-Music Generation Pipeline Automation
# Usage: ./run_pipeline.sh <input_midi> <emotion>
# Example: ./run_pipeline.sh test_melody.mid Q1
# ==========================================

# 1. 引数のチェック
INPUT_MIDI=$1
EMOTION=$2

if [ -z "$INPUT_MIDI" ] || [ -z "$EMOTION" ]; then
    echo "エラー: 引数が足りません。"
    echo "使用法: ./run_pipeline.sh <midiファイルパス> <感情ラベル>"
    echo "例: ./run_pipeline.sh test_melody.mid Q1"
    exit 1
fi

# 2. パスの設定（絶対パスに変換して、ディレクトリ移動しても迷子にならないようにする）
# realpathコマンドがない環境(一部のMac等)では $(pwd)/filename などを使う必要がありますが、UbuntuならこれでOK
ABS_MIDI_PATH=$(realpath "$INPUT_MIDI")
PROJECT_ROOT=$(pwd)

HARMONIZER_DIR="$PROJECT_ROOT/EMO_Harmonizer"
STAGE2_DIR="$PROJECT_ROOT/EMO-Disentanger"

# 一時出力先（Harmonizerの結果）
TEMP_OUT_DIR="generation/pipeline_temp"
# 最終出力先（Stage2の結果）
FINAL_OUT_DIR="generation/pipeline_output"

echo "=========================================="
echo "Pipeline Start"
echo "Input: $ABS_MIDI_PATH"
echo "Emotion: $EMOTION"
echo "=========================================="

# ------------------------------------------------
# Step 1: EMO_Harmonizer で和声を生成
# ------------------------------------------------
echo "[Step 1] Running EMO_Harmonizer..."

cd "$HARMONIZER_DIR" || exit

# 出力先をクリーニング（前の結果が混ざらないように）
rm -rf "$TEMP_OUT_DIR"
mkdir -p "$TEMP_OUT_DIR"

# user_harmonize.py を実行
# ※キーは user_harmonize.py 側で自動決定されるため、ダミーまたは省略でOKですが
#   スクリプトの引数仕様に合わせて念のため渡しておきます（内部では無視されます）
python user_harmonize.py \
  --input_midi "$ABS_MIDI_PATH" \
  --emotion "$EMOTION" \
  --output_dir "$TEMP_OUT_DIR" \
  --key C  # 評価実験用ロジックにより、内部で自動的に C or a に上書きされます

if [ $? -ne 0 ]; then
    echo "[Error] Step 1 failed."
    exit 1
fi

# ------------------------------------------------
# Step 2: データを Stage 2 へ渡す
# ------------------------------------------------
echo "[Step 2] Transferring data..."

# Stage2 側のディレクトリ準備
cd "$STAGE2_DIR" || exit
mkdir -p "$FINAL_OUT_DIR"

# Harmonizerが出力した roman.txt をコピー
# パスは ../EMO_Harmonizer/generation/pipeline_temp/*.txt となる
cp "$HARMONIZER_DIR/$TEMP_OUT_DIR"/*_roman.txt "$FINAL_OUT_DIR/"

if [ $? -ne 0 ]; then
    echo "[Error] Step 2 failed (File copy error)."
    exit 1
fi

echo "Data copied to: $STAGE2_DIR/$FINAL_OUT_DIR"

# ------------------------------------------------
# Step 3: Stage 2 で伴奏を生成
# ------------------------------------------------
echo "[Step 3] Running Stage 2 Accompaniment..."

# inference.py を実行
# 設定（GPT-2, temp=1.0, top_p=0.9 はコード内で固定済み）を使用
python stage2_accompaniment/inference.py \
  -m gpt2 \
  -c stage2_accompaniment/config/emopia_finetune_gpt2.yaml \
  -r functional \
  -o "$FINAL_OUT_DIR" \
  -i best_weight/Functional-two/emopia_acccompaniment_finetune_gpt2/ep300_loss0.120_params.pt

if [ $? -ne 0 ]; then
    echo "[Error] Step 3 failed."
    exit 1
fi

echo "=========================================="
echo "Pipeline Completed Successfully!"
echo "Result saved in: $STAGE2_DIR/$FINAL_OUT_DIR"
echo "=========================================="
