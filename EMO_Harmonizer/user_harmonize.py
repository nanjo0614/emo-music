import argparse
import os
import sys
from typing import List, Tuple
from itertools import chain

import yaml
import torch

# EMO_Harmonizer 内のモジュールを import
from dataloader import REMISkylineToMidiTransformerDataset
from model.music_performer import MusicPerformer
from convert2midi import event_to_midi, TempoEvent
from utils import pickle_load

from inference_user import (
    load_melody_notes,
    estimate_last_bar,
    quantize_notes,
    build_melody_events,
)

import inference as emo_inference  # sampling utilities, generate_conditional, events2bars, event_to_txt


def build_melody_event_bars_from_midi(
    midi_path: str,
    key: str,
    event2idx,
) -> List[List[int]]:
    """
    ユーザ MIDI からメロディを抽出し、EMO_Harmonizer の generate_conditional 用
    「bar ごとのメロディイベント (ID 列)」を作成する。
    """
    # 1. MIDI → ノート列
    midi, notes = load_melody_notes(midi_path)
    if not notes:
        raise RuntimeError(f"メロディノートが見つかりません: {midi_path}")

    # 2. ノート列を EMOPIA と同じ 16 分音符グリッドに量子化
    last_bar = estimate_last_bar(notes)
    quantized_notes = quantize_notes(notes, last_bar)

    # 3. functional / relative_melody=True 想定でイベント列を生成
    #    ★ ここで渡す key によって、同じピッチでも度数(Degree)の解釈が変わる
    events: List[str] = build_melody_events(
        quantized_notes,
        key=key,
        emotion=None,
        add_track_tokens=True,
        add_eos=True,
    )

    # 4. Prefix / Suffix を削除して、Bar_None ごとに分割
    tokens = list(events)

    # 先頭の Emotion / Key / Track_Melody を除去
    if tokens and tokens[0].startswith("Emotion_"):
        tokens = tokens[1:]
    if tokens and tokens[0].startswith("Key_"):
        tokens = tokens[1:]
    if tokens and tokens[0] == "Track_Melody":
        tokens = tokens[1:]

    # 末尾 EOS_None を除去
    if tokens and tokens[-1] == "EOS_None":
        tokens = tokens[:-1]

    # Bar_None で bar ごとのイベント列に分割
    bar_indices = [i for i, t in enumerate(tokens) if t == "Bar_None"]
    if not bar_indices:
        raise RuntimeError("メロディイベント内に 'Bar_None' が存在しません。")

    bar_indices.append(len(tokens))

    melody_events: List[List[int]] = []
    for b in range(len(bar_indices) - 1):
        start = bar_indices[b]
        end = bar_indices[b + 1]
        bar_tokens = tokens[start:end]

        bar_ids: List[int] = []
        for tk in bar_tokens:
            if tk not in event2idx:
                continue
            bar_ids.append(event2idx[tk])

        if bar_ids:
            melody_events.append(bar_ids)

    if not melody_events:
        raise RuntimeError("有効なメロディイベントが 1 bar も生成できませんでした。")

    return melody_events


def map_user_emotion_to_harmonizer_token(user_emotion: str) -> str:
    """
    ユーザ指定の感情ラベル → EMO_Harmonizer が使う Emotion_* トークンに変換。
    """
    user_emotion = user_emotion.strip()

    if user_emotion in ["Positive", "Negative"]:
        return f"Emotion_{user_emotion}"

    if user_emotion in ["Q1", "Q4"]:
        return "Emotion_Positive"
    if user_emotion in ["Q2", "Q3"]:
        return "Emotion_Negative"

    raise ValueError(
        f"未知の emotion 指定です: {user_emotion} "
        f"(使用可能: Positive, Negative, Q1, Q2, Q3, Q4)"
    )


def determine_target_key(emotion_label: str) -> str:
    """
    評価実験用ロジック:
    入力メロディは Cメジャースケール(平行調Am) 固定とする。
    感情ラベルに応じて、強制的に出力キーを決定する。
    
    Q1, Q4 (Positive系) -> C Major ('C')
    Q2, Q3 (Negative系) -> A Minor ('a')
    """
    e = emotion_label.strip()
    
    # Positive系 -> C Major
    if e in ['Q1', 'Q4', 'Positive']:
        return 'C'
    
    # Negative系 -> A Minor (小文字の 'a')
    if e in ['Q2', 'Q3', 'Negative']:
        return 'a'
        
    # フォールバック (デフォルト C)
    print(f"[Warn] Emotion '{e}' unknown for key rule. Defaulting to Key C.")
    return 'C'


def main():
    parser = argparse.ArgumentParser(description="ユーザメロディ + 感情ラベルから EMO_Harmonizer で和声付きリードシートを生成するスクリプト")
    parser.add_argument(
        "--input_midi",
        required=True,
        help="ユーザが作成したメロディの MIDI ファイルパス (Cメジャー/Aマイナー前提)",
    )
    parser.add_argument(
        "--emotion",
        required=True,
        help="感情ラベル: Positive / Negative / Q1 / Q2 / Q3 / Q4",
    )
    # 以前は --key が必須でしたが、自動決定するため任意(または無視)にします
    parser.add_argument(
        "--key",
        default="C",
        help="（評価実験モードでは無視されます）",
    )
    parser.add_argument(
        "-c",
        "--configuration",
        default="config/emopia_finetune.yaml",
        help="EMO_Harmonizer の学習時と同じ YAML 設定ファイル",
    )
    parser.add_argument(
        "-r",
        "--representation",
        default="functional",
        choices=["functional"],
        help="現状 functional のみサポート",
    )
    parser.add_argument(
        "-k",
        "--key_determine",
        default="none",
        choices=["rule", "none"],
        help="none推奨 (キーはemotionから自動決定するため)",
    )
    parser.add_argument(
        "-i",
        "--inference_params",
        default="emo_harmonizer_ckpt_functional/best_params.pt",
        help="EMO_Harmonizer の学習済みパラメータ (.pt)",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default="generation/user_harmonize",
        help="生成結果を保存するディレクトリ",
    )
    parser.add_argument(
        "--max_bars",
        type=int,
        default=128,
        help="生成する最大小節数",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=1.1,
        help="サンプリング温度 (temperature)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.99,
        help="nucleus sampling の top_p",
    )
    parser.add_argument(
        "--no_midi",
        action="store_true",
        help="MIDI 出力を行わない場合に指定",
    )
    parser.add_argument(
        "--play_midi",
        action="store_true",
        help="FluidSynth で wav も同時にレンダリングする場合に指定",
    )

    args = parser.parse_args()

    # 0. 感情ラベルを EMO_Harmonizer のトークンにマッピング
    emotion_token = map_user_emotion_to_harmonizer_token(args.emotion)

    # === 評価実験用: キーの強制決定 ===
    # ユーザ入力に関わらず、Emotionに合わせてキーを固定
    target_key = determine_target_key(args.emotion)
    print(f"[Experiment Mode] Emotion: {args.emotion} -> Force Key: {target_key}")

    # 1. YAML コンフィグ読み込み
    train_conf_path = args.configuration
    if not os.path.exists(train_conf_path):
        raise FileNotFoundError(f"configuration が存在しません: {train_conf_path}")

    train_conf = yaml.load(open(train_conf_path, "r"), Loader=yaml.FullLoader)
    representation = args.representation

    # functional 前提
    relative_chord = True
    relative_melody = True

    # 2. デバイス設定
    gpuid = train_conf["training"]["gpuid"]
    use_cuda = torch.cuda.is_available() and gpuid >= 0
    if use_cuda:
        torch.cuda.set_device(gpuid)

    device = torch.device(f"cuda:{gpuid}" if use_cuda else "cpu")
    print(f"[info] device = {device}")

    # 3. データローダ (vocab 取得用にだけ使う)
    data_path = train_conf["data_loader"]["data_path"].format(representation)
    vocab_path = train_conf["data_loader"]["vocab_path"].format(representation)
    val_split = train_conf["data_loader"]["val_split"]

    dset = REMISkylineToMidiTransformerDataset(
        data_dir=data_path,
        vocab_file=vocab_path,
        model_dec_seqlen=train_conf["model"]["max_len"],
        pieces=pickle_load(val_split),
        pad_to_same=True,
        predict_key=False,
    )

    # 4. モデル構築 & 重みロード
    model_conf = train_conf["model"]

    model = MusicPerformer(
        dset.vocab_size,
        model_conf["n_layer"],
        model_conf["n_head"],
        model_conf["d_model"],
        model_conf["d_ff"],
        model_conf["d_embed"],
        use_segment_emb=model_conf["use_segemb"],
        n_segment_types=model_conf["n_segment_types"],
        favor_feature_dims=model_conf["feature_map"]["n_dims"],
        use_chord_mhot_emb=False,
    ).to(device)

    if not os.path.exists(args.inference_params):
        raise FileNotFoundError(f"inference_params が存在しません: {args.inference_params}")

    pretrained_dict = torch.load(args.inference_params, map_location=device)
    pretrained_dict = {
        k: v for k, v in pretrained_dict.items() if "feature_map.omega" not in k
    }
    model_state_dict = model.state_dict()
    model_state_dict.update(pretrained_dict)
    model.load_state_dict(model_state_dict)
    model.eval()
    print("[info] EMO_Harmonizer モデルをロードしました")

    # 生成時に inference.py の generate_conditional が参照するグローバル変数
    emo_inference.max_dec_inp_len = 1024

    # 5. ユーザメロディ → bar ごとのメロディイベント (ID 列)
    # ここで target_key を渡すことで、入力MIDI(C scale)を
    # 'C' (Major) あるいは 'a' (Minor) の度数として解釈させる
    melody_events = build_melody_event_bars_from_midi(
        args.input_midi,
        key=target_key,
        event2idx=dset.event2idx,
    )
    print(f"[info] #bars (from melody) = {len(melody_events)}")

    max_bars = min(args.max_bars, len(melody_events))

    # 6. Emotion / Key トークン列を組み立て
    if emotion_token not in dset.event2idx:
        raise KeyError(f"vocabulary に存在しない Emotion トークンです: {emotion_token}")

    key_token = f"Key_{target_key}"
    if relative_chord:
        if key_token not in dset.event2idx:
            raise KeyError(f"vocabulary に存在しない Key トークンです: {key_token}")
        generated = [dset.event2idx[emotion_token], dset.event2idx[key_token]]
        seg_inp = [0, 0]
    else:
        generated = [dset.event2idx[emotion_token]]
        seg_inp = [0]

    print(f"[info] prefix tokens: {[emotion_token, key_token]}")

    # 7. コード (和声) 生成
    with torch.no_grad():
        generated_ids = emo_inference.generate_conditional(
            model,
            dset.event2idx,
            dset.idx2event,
            melody_events,
            generated,
            seg_inp,
            max_bars=max_bars,
            temp=args.temp,
            top_p=args.top_p,
            inadmissibles=None,
        )

    # 8. イベント列に戻す
    generated_events = emo_inference.word2event(generated_ids, dset.idx2event)

    # 先頭は Emotion_* のはずなので、Roman 版ではこれを落とす
    events_roman = generated_events[1:]

    # MIDI 用に絶対音高へ変換 (functional / relative_melody=True 前提)
    events_abs, lead_sheet_bars = emo_inference.events2bars(
        key_token, generated_events, relative_melody=relative_melody
    )

    # 9. ファイル出力
    os.makedirs(args.output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(args.input_midi))[0]
    base_stub = f"harm_{basename}_{args.emotion}"

    # (1) roman (functional) イベント列
    roman_txt_path = os.path.join(args.output_dir, base_stub + "_roman.txt")
    emo_inference.event_to_txt(events_roman, output_event_path=roman_txt_path)
    print(f"[info] wrote roman events to {roman_txt_path}")

    # (2) 絶対音高版イベント列
    abs_txt_path = os.path.join(args.output_dir, base_stub + ".txt")
    emo_inference.event_to_txt(events_abs, output_event_path=abs_txt_path)

    # (3) MIDI
    if not args.no_midi:
        midi_path = os.path.join(args.output_dir, base_stub + ".mid")
        event_to_midi(
            key_token,
            list(chain(*lead_sheet_bars[:max_bars])),
            mode="skyline",
            play_chords=True,
            enforce_tempo=True,
            enforce_tempo_evs=[TempoEvent(110, 0, 0)],
            output_midi_path=midi_path,
        )
        print(f"[info] wrote MIDI to {midi_path}")

        if args.play_midi:
            try:
                from midi2audio import FluidSynth
                wav_path = os.path.join(args.output_dir, base_stub + ".wav")
                sound_font_path = "SalamanderGrandPiano-SF2-V3+20200602/SalamanderGrandPiano-V3+20200602.sf2"
                fs = FluidSynth(sound_font_path)
                fs.midi_to_audio(midi_path, wav_path)
                print(f"[info] wrote WAV to {wav_path}")
            except ImportError:
                print("[warn] midi2audio / FluidSynth not found, skipping wav.")

if __name__ == "__main__":
    main()