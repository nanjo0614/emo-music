import argparse
import os
from typing import List, Tuple

import miditoolkit

# Reuse EMOPIA key/degree logic
try:
    # このスクリプトを EMO_Harmonizer 直下に置いて実行することを想定
    from representations.convert_key import pitch2degree, MAJOR_KEY, MINOR_KEY  # type: ignore
except ImportError:
    # EMO_Harmonizer をパッケージとして import した場合のフォールバック
    from EMO_Harmonizer.representations.convert_key import pitch2degree, MAJOR_KEY, MINOR_KEY  # type: ignore

# === Timing resolution (EMOPIA / EMO-Harmonizer と揃える) ===
BEAT_RESOL = 480          # ticks per beat (quarter note)
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4  # 16th-note grid (120 ticks)


def quantize_tick(t: int) -> int:
    """生の tick 値を TICK_RESOL グリッドに最も近い位置へ量子化."""
    if t < 0:
        t = 0
    q = int(round(t / TICK_RESOL)) * TICK_RESOL
    if q < 0:
        q = 0
    return q


def _normalize_ticks_per_beat(midi: "miditoolkit.MidiFile") -> None:
    """
    EMOPIA / EMO-Harmonizer は 480 ticks/beat 前提。
    ユーザー MIDI の ticks_per_beat が異なる場合は、
    480 に合うように全イベントの時刻をスケーリングする（破壊的変更）。
    """
    tpb = midi.ticks_per_beat
    if tpb == BEAT_RESOL:
        return

    factor = BEAT_RESOL / float(tpb)

    # notes
    for inst in midi.instruments:
        for n in inst.notes:
            n.start = int(round(n.start * factor))
            n.end = int(round(n.end * factor))

    # tempo changes
    # （これはすでに動いているのでそのまま）
    for tempo in getattr(midi, "tempo_changes", []):
        tempo.time = int(round(tempo.time * factor))

    # time signatures
    # miditoolkit のバージョンによって属性名が違うので両方に対応する
    ts_list = getattr(midi, "time_signature_changes", None)
    if ts_list is None:
        # 古いバージョン向けフォールバック
        ts_list = getattr(midi, "time_signatures", [])

    for ts in ts_list:
        ts.time = int(round(ts.time * factor))

    # 最後に ticks_per_beat を更新
    midi.ticks_per_beat = BEAT_RESOL



def load_melody_notes(midi_path: str):
    """
    ユーザーメロディを MIDI から読み込む:
      - ticks_per_beat を 480 に正規化
      - 最初の non-drum トラックをメロディとみなす
      - monophonic 化のため、オーバーラップをカット
    """
    midi = miditoolkit.midi.parser.MidiFile(midi_path)

    # TPB 正規化
    _normalize_ticks_per_beat(midi)

    if not midi.instruments:
        raise ValueError(f"No instruments found in MIDI: {midi_path}")

    # できるだけ non-drum トラックを優先
    inst = None
    for track in midi.instruments:
        if not track.is_drum:
            inst = track
            break
    if inst is None:
        inst = midi.instruments[0]

    notes = sorted(inst.notes, key=lambda n: (n.start, n.pitch))

    if not notes:
        raise ValueError(f"No notes found in melody track: {midi_path}")

    # monophonic 化：次のノートとオーバーラップしていたら終端を切る
    for i in range(len(notes) - 1):
        if notes[i].end > notes[i + 1].start:
            notes[i].end = notes[i + 1].start

    return midi, notes


def quantize_notes(notes, last_bar: int):
    """
    ノートを EMOPIA グリッドに量子化する:
      - start: TICK_RESOL に丸め
      - duration: max(TICK_RESOL, q_end - q_start)
      - 曲末(last_bar)以降にはみ出したノートは切る or 破棄
    戻り値: [(start_tick, duration, pitch), ...]
    """
    quantized = []
    song_end_tick = last_bar * BAR_RESOL

    for n in notes:
        q_start = quantize_tick(n.start)
        q_end = quantize_tick(n.end)

        dur = max(TICK_RESOL, q_end - q_start)

        # 完全に曲末より後ろは無視
        if q_start >= song_end_tick:
            continue

        # はみ出し分は曲末にクランプ
        if q_start + dur > song_end_tick:
            dur = song_end_tick - q_start
            if dur <= 0:
                continue

        quantized.append((q_start, dur, n.pitch))

    quantized.sort(key=lambda x: (x[0], x[2]))
    return quantized


def estimate_last_bar(notes) -> int:
    """
    全ノートを含むのに必要な小節数を推定。
    """
    last_end = max(n.end for n in notes)
    last_bar = max(1, int((last_end + BAR_RESOL - 1) // BAR_RESOL))
    return last_bar


def build_melody_events(
    quantized_notes: List[Tuple[int, int, int]],
    key: str,
    emotion: str = None,
    add_track_tokens: bool = True,
    add_eos: bool = True,
) -> List[str]:
    """
    量子化済みノート列から EMOPIA 形式のメロディイベント列を構築する。
    （functional, relative_melody=True 想定）

    形式:
        [Emotion_*]
        Key_<key>
        Track_Melody
        Bar_None
        Beat_<idx>
        Note_Octave_<octave>
        Note_Degree_<roman>
        Note_Duration_<ticks>
        ...
        EOS_None
    """
    key = key.strip()
    if key not in MAJOR_KEY and key not in MINOR_KEY:
        raise ValueError(
            f"Unsupported key '{key}'. "
            f"Use one of: {list(MAJOR_KEY) + list(MINOR_KEY)}"
        )

    events: List[str] = []

    # Emotion（任意）
    if emotion is not None:
        # "Positive" -> "Emotion_Positive" のように正規化
        emotion_token = emotion if emotion.startswith("Emotion_") else f"Emotion_{emotion}"
        events.append(emotion_token)

    # Key
    events.append(f"Key_{key}")

    if add_track_tokens:
        events.append("Track_Melody")

    if not quantized_notes:
        if add_eos:
            events.append("EOS_None")
        return events

    # bar ごとにグルーピング
    last_tick = max(s + d for (s, d, _) in quantized_notes)
    n_bars = max(1, (last_tick + BAR_RESOL - 1) // BAR_RESOL)

    idx = 0
    for bar_idx in range(n_bars):
        bar_start = bar_idx * BAR_RESOL
        bar_end = bar_start + BAR_RESOL

        events.append("Bar_None")

        while idx < len(quantized_notes):
            start, dur, pitch = quantized_notes[idx]
            if start >= bar_end:
                break
            if not (bar_start <= start < bar_end):
                idx += 1
                continue

            # 小節内の 16 分音符インデックス (0–15)
            beat = (start - bar_start) // TICK_RESOL

            # ピッチ -> (オクターブ, ローマ数字度数)
            octave, roman = pitch2degree(key, pitch)

            events.append(f"Beat_{beat}")
            events.append(f"Note_Octave_{octave}")
            events.append(f"Note_Degree_{roman}")
            events.append(f"Note_Duration_{dur}")

            idx += 1

    if add_eos:
        events.append("EOS_None")

    return events


def write_events_txt(events: List[str], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(str(ev).strip() + "\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "ユーザーのメロディ MIDI を EMOPIA 形式のメロディイベント "
            "(functional, relative_melody=True) に変換するスクリプト"
        )
    )
    parser.add_argument(
        "--input_midi", "-i", required=True,
        help="入力メロディ MIDI ファイルへのパス"
    )
    parser.add_argument(
        "--output_txt", "-o", required=True,
        help="出力イベントテキストファイルへのパス"
    )
    parser.add_argument(
        "--key", "-k", default="C",
        help=(
            "メロディの調（例: C, G#, f など）。"
            "EMOPIA の MAJOR_KEY / MINOR_KEY に含まれる表記である必要があります。"
        ),
    )
    parser.add_argument(
        "--emotion", "-e", default=None,
        help=(
            "任意の感情タグ（例: Positive / Negative / Q1 / Q2 / Q3 / Q4）。"
            "ファイル上は 'Emotion_<tag>' というトークンになります。"
        ),
    )

    args = parser.parse_args()

    midi, notes = load_melody_notes(args.input_midi)
    last_bar = estimate_last_bar(notes)
    q_notes = quantize_notes(notes, last_bar)
    events = build_melody_events(q_notes, key=args.key, emotion=args.emotion)

    write_events_txt(events, args.output_txt)

    print(f"[info] Parsed melody: {len(q_notes)} notes, {len(events)} events")
    print(f"[info] Written events to: {args.output_txt}")


if __name__ == "__main__":
    main()
