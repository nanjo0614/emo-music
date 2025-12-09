import os
import sys
import time
import yaml
import shutil
import argparse
import numpy as np
from itertools import chain
from collections import defaultdict
import torch

from dataloader import REMISkylineToMidiTransformerDataset, pickle_load
from convert2midi import event_to_midi
from convert_key import degree2pitch, roman2majorDegree, roman2minorDegree

sys.path.append('./model')

max_bars = 16
max_dec_inp_len = 2048

emotion_events = ['Emotion_Q1', 'Emotion_Q2', 'Emotion_Q3', 'Emotion_Q4']
samp_per_piece = 1

major_map = [0, 4, 7]
minor_map = [0, 3, 7]
diminished_map = [0, 3, 6]
augmented_map = [0, 4, 8]
dominant_map = [0, 4, 7, 10]
major_seventh_map = [0, 4, 7, 11]
minor_seventh_map = [0, 3, 7, 10]
diminished_seventh_map = [0, 3, 6, 9]
half_diminished_seventh_map = [0, 3, 6, 10]
sus_2_map = [0, 2, 7]
sus_4_map = [0, 5, 7]

chord_maps = {
    'M': major_map,
    'm': minor_map,
    'o': diminished_map,
    '+': augmented_map,
    '7': dominant_map,
    'M7': major_seventh_map,
    'm7': minor_seventh_map,
    'o7': diminished_seventh_map,
    '/o7': half_diminished_seventh_map,
    'sus2': sus_2_map,
    'sus4': sus_4_map
}
chord_maps = {k: np.array(v) for k, v in chord_maps.items()}

DEFAULT_SCALE = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B']
MAJOR_KEY = np.array(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
MINOR_KEY = np.array(['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b'])


###############################################
# sampling utilities
###############################################
def construct_inadmissible_set(tempo_val, event2idx, tolerance=20):
    inadmissibles = []

    for k, i in event2idx.items():
        if 'Tempo' in k and 'Conti' not in k and abs(int(k.split('_')[-1]) - tempo_val) > tolerance:
            inadmissibles.append(i)

    print(inadmissibles)

    return np.array(inadmissibles)


def temperature(logits, temperature, inadmissibles=12):
    if inadmissibles is not None:
        logits[inadmissibles] -= np.inf

    try:
        probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        assert np.count_nonzero(np.isnan(probs)) == 0
    except:
        print('overflow detected, use 128-bit')
        logits = logits.astype(np.float128)
        probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        probs = probs.astype(float)
    return probs


def nucleus(probs, p):
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][1]
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:3]  # just assign a value
    candi_probs = np.array([probs[i] for i in candi_index], dtype=np.float64)
    candi_probs /= sum(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word


##############################################
# data manipulation utilities
##############################################
def merge_tracks(melody_track, chord_track):
    events = melody_track[1:3]

    melody_beat = defaultdict(list)
    if len(melody_track) > 3:
        note_seq = []
        beat = melody_track[3]
        melody_track = melody_track[4:]
        for p in range(len(melody_track)):
            if 'Beat' in melody_track[p]:
                melody_beat[beat] = note_seq
                note_seq = []
                beat = melody_track[p]
            else:
                note_seq.append(melody_track[p])
        melody_beat[beat] = note_seq

    chord_beat = defaultdict(list)
    if len(chord_track) > 2:
        chord_seq = []
        beat = chord_track[2]
        chord_track = chord_track[3:]
        for p in range(len(chord_track)):
            if 'Beat' in chord_track[p]:
                chord_beat[beat] = chord_seq
                chord_seq = []
                beat = chord_track[p]
            else:
                chord_seq.append(chord_track[p])
        chord_beat[beat] = chord_seq

    for b in range(16):
        beat = 'Beat_{}'.format(b)
        if beat in chord_beat or beat in melody_beat:
            events.append(beat)
            if beat in chord_beat:
                events.extend(chord_beat[beat])
            if beat in melody_beat:
                events.extend(melody_beat[beat])

    return events


def read_generated_events(events_file, event2idx):
    events = open(events_file).read().splitlines()
    if 'Key' in events[0]:
        key = events[0]
    else:
        key = 'Key_C'

    bar_pos = np.where(np.array(events) == 'Bar_None')[0].tolist()
    bar_pos.append(len(events))

    raw_bars = []
    
    # 欠落トークンのログ収集用
    unknown_tokens_count = defaultdict(int)

    for b in range(len(bar_pos)-1):
        bar_events = events[bar_pos[b]: bar_pos[b+1]]
        filtered_idx_seq = []
        for e in bar_events:
            # event2idx に存在しないトークンは捨てるが、ログに残す
            if e not in event2idx:
                unknown_tokens_count[e] += 1
                continue
            filtered_idx_seq.append(event2idx[e])
        raw_bars.append(filtered_idx_seq)

    # 警告出力
    if unknown_tokens_count:
        print(f"\n[WARNING] {os.path.basename(events_file)} から以下のトークンが欠落しました（Vocab未定義）:")
        for token, count in unknown_tokens_count.items():
            print(f"  - {token}: {count} 個")
        print("  -> Chord_... や Note_... が含まれている場合は要注意です。\n")

    # === マージ処理: [Melody_Bar1, Chord_Bar1, Melody_Bar2, Chord_Bar2 ...] を結合する ===
    lead_sheet_bars = []
    
    # Harmonizer出力は メロディ小節 -> コード小節 の順で交互に来るため、2つずつ結合する
    # 端数が出る場合(最後の小節など)は安全策としてそのまま追加
    for i in range(0, len(raw_bars), 2):
        if i + 1 >= len(raw_bars):
            lead_sheet_bars.append(raw_bars[i])
            break
            
        melody_part = raw_bars[i]
        chord_part = raw_bars[i+1]
        
        # Chordパートの先頭にある Bar_None (ID) を削除して結合
        bar_none_id = event2idx.get('Bar_None')
        if chord_part and chord_part[0] == bar_none_id:
            chord_part = chord_part[1:]
            
        merged_bar = melody_part + chord_part
        lead_sheet_bars.append(merged_bar)

    return key, lead_sheet_bars


def word2event(word_seq, idx2event):
    return [idx2event[w] for w in word_seq]


def extract_midi_events_from_generation(key, events, relative_melody=False):
    """
    key: 'Key_C' など
    events: イベント列（文字列）
    relative_melody: True の場合、(Note_Octave, Note_Degree) → Note_Pitch に変換
    """
    if relative_melody:
        new_events = []
        keyname = key.split('_')[1]
        
        # オクターブ状態の保持（メロディ欠落対策）
        current_octave = 4
        roman = None

        for evs in events:
            # Note_Octave を見つけたら更新、見つからなくても前の値を維持
            if 'Note_Octave' in evs:
                parts = evs.split('_')
                if len(parts) >= 3:
                    try:
                        current_octave = int(parts[2])
                    except Exception:
                        pass # パース失敗時は維持

            elif 'Note_Degree' in evs:
                parts = evs.split('_')
                if len(parts) < 3:
                    continue
                roman = parts[2]
                
                # 度数→ピッチ変換（エラーが出ても無視して続行）
                try:
                    pitch = degree2pitch(keyname, current_octave, roman)
                    pitch = max(21, min(108, pitch)) # Clamp
                    new_events.append('Note_Pitch_{}'.format(pitch))
                except Exception:
                    # キー変換エラー等の場合はスキップ
                    pass

            elif 'Chord_' in evs:
                if 'None' in evs or 'Conti' in evs:
                    new_events.append(evs)
                else:
                    try:
                        root, quality = evs.split('_')[1], evs.split('_')[2]
                        if keyname in MAJOR_KEY:
                            root = roman2majorDegree[root]
                        else:
                            root = roman2minorDegree[root]
                        new_events.append('Chord_{}_{}'.format(root, quality))
                    except:
                        continue
            else:
                new_events.append(evs)

        events = new_events

    # Track分割（Full Trackのみ抽出）
    lead_sheet_starts = np.where(np.array(events) == 'Track_LeadSheet')[0].tolist()
    full_starts = np.where(np.array(events) == 'Track_Full')[0].tolist()

    midi_bars = []
    # Full Track部分を抽出してBarごとに区切る
    # Track_Full から 次の Track_LeadSheet (または末尾) まで
    for st, ed in zip(full_starts, lead_sheet_starts[1:] + [len(events)]):
        bar_segment = events[st + 1: ed]
        
        # Bar_None で区切ってリスト化
        current_bar_content = []
        for ev in bar_segment:
            if ev == 'Bar_None':
                if current_bar_content:
                    midi_bars.append(current_bar_content)
                current_bar_content = [ev]
            else:
                current_bar_content.append(ev)
        if current_bar_content:
            midi_bars.append(current_bar_content)

    return midi_bars


def get_position_idx(event):
    return int(event.split('_')[-1])


def event_to_txt(events, output_event_path):
    f = open(output_event_path, 'w')
    print(*events, sep='\n', file=f)


def midi_to_wav(midi_path, output_path):
    sound_font_path = 'SalamanderGrandPiano-SF2-V3+20200602/SalamanderGrandPiano-V3+20200602.sf2'
    from midi2audio import FluidSynth
    fs = FluidSynth(sound_font_path)
    fs.midi_to_audio(midi_path, output_path)


################################################
# main generation function
################################################
def generate_conditional(model, event2idx, idx2event, lead_sheet_events, primer,
                         max_events=3000, skip_check=False, max_bars=None,
                         temp=1.2, top_p=0.9, inadmissibles=None,
                         model_type="performer"):
    generated = primer + [event2idx['Track_LeadSheet']] + lead_sheet_events[0] + [event2idx['Track_Full']]
    seg_inp = [0 for _ in range(len(generated))]
    seg_inp[-1] = 1

    target_bars, generated_bars = len(lead_sheet_events), 0
    if max_bars is not None:
        target_bars = min(max_bars, target_bars)

    steps = 0
    time_st = time.time()
    cur_pos = 0
    failed_cnt = 0

    while generated_bars < target_bars:
        if len(generated) < max_dec_inp_len:
            dec_input = torch.tensor([generated]).long().to(next(model.parameters()).device)
            dec_seg_inp = torch.tensor([seg_inp]).long().to(next(model.parameters()).device)
        else:
            dec_input = torch.tensor([generated[-max_dec_inp_len:]]).long().to(next(model.parameters()).device)
            dec_seg_inp = torch.tensor([seg_inp[-max_dec_inp_len:]]).long().to(next(model.parameters()).device)

        if model_type == "performer":
            logits = model(
                dec_input,
                seg_inp=dec_seg_inp,
                keep_last_only=True,
                attn_kwargs={'omit_feature_map_draw': steps > 0}
            )
        else:
            logits = model(
                dec_input,
                seg_inp=dec_seg_inp,
                keep_last_only=True,
            )

        logits = (logits[0]).cpu().detach().numpy()
        probs = temperature(logits, temp, inadmissibles=inadmissibles)
        word = nucleus(probs, top_p)
        word_event = idx2event[word]

        if not skip_check:
            if 'Beat' in word_event:
                event_pos = get_position_idx(word_event)
                if not event_pos >= cur_pos:
                    failed_cnt += 1
                    if failed_cnt >= 256:
                        print('[FATAL] model stuck, exiting with generated events ...')
                        return generated
                    continue
                else:
                    cur_pos = event_pos
                    failed_cnt = 0

        if word_event == 'Track_LeadSheet':
            steps += 1
            generated.append(word)
            seg_inp.append(0)
            generated_bars += 1
            print('[info] generated {} bars, #events = {}'.format(generated_bars, len(generated)))

            if generated_bars < target_bars:
                generated.extend(lead_sheet_events[generated_bars])
                seg_inp.extend([0 for _ in range(len(lead_sheet_events[generated_bars]))])

                generated.append(event2idx['Track_Full'])
                seg_inp.append(1)
                cur_pos = 0
            continue

        if word_event == 'PAD_None' or (word_event == 'EOS_None' and generated_bars < target_bars - 1):
            continue
        elif word_event == 'EOS_None' and generated_bars == target_bars - 1:
            print('[info] gotten eos')
            generated.append(word)
            break

        generated.append(word)
        seg_inp.append(1)
        steps += 1

        if len(generated) > max_events:
            print('[info] max events reached')
            break

    print('-- generated events:', len(generated))
    print('-- time elapsed  : {:.2f} secs'.format(time.time() - time_st))
    print('-- time per event: {:.2f} secs'.format((time.time() - time_st) / len(generated)))
    return generated[:-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    required = parser.add_argument_group('required arguments')
    required.add_argument('-m', '--model_type',
                          choices=['performer', 'gpt2'],
                          help='model backbone', required=True)
    required.add_argument('-c', '--configuration',
                          help='configurations of training', required=True)
    required.add_argument('-r', '--representation',
                          choices=['remi', 'functional'],
                          help='representation for symbolic music', required=True)
    parser.add_argument('-i', '--inference_params',
                        help='inference parameters')
    parser.add_argument('-o', '--output_dir',
                        default='generation/emopia_functional_two',
                        help='output directory')
    parser.add_argument('-p', '--play_midi',
                        default=False,
                        help='play midi to audio using FluidSynth', action='store_true')
    args = parser.parse_args()

    train_conf_path = args.configuration
    train_conf = yaml.load(open(train_conf_path, 'r'), Loader=yaml.FullLoader)
    print(train_conf)

    representation = args.representation
    if representation == 'remi':
        relative_melody = False
    elif representation == 'functional':
        relative_melody = True

    inference_param_path = args.inference_params
    gen_leadsheet_dir = args.output_dir
    play_midi = args.play_midi

    train_conf_ = train_conf['training']
    gpuid = train_conf_['gpuid']
    if torch.cuda.is_available():
        torch.cuda.set_device(gpuid)
    else:
        gpuid = 'cpu'

    val_split = train_conf['data_loader']['val_split']
    dset = REMISkylineToMidiTransformerDataset(
        train_conf['data_loader']['data_path'].format(representation),
        train_conf['data_loader']['vocab_path'].format(representation),
        model_dec_seqlen=train_conf['model']['max_len'],
        pieces=pickle_load(val_split),
        pad_to_same=True,
    )

    model_conf = train_conf['model']
    model_type = args.model_type
    if model_type == "performer":
        from model.music_performer import MusicPerformer

        model = MusicPerformer(
            dset.vocab_size, model_conf['n_layer'], model_conf['n_head'],
            model_conf['d_model'], model_conf['d_ff'], model_conf['d_embed'],
            use_segment_emb=model_conf['use_segemb'], n_segment_types=model_conf['n_segment_types'],
            favor_feature_dims=model_conf['feature_map']['n_dims']
        )
        if gpuid != 'cpu': model = model.cuda(gpuid)
        temp, top_p = 1.0, 0.9 # Optimized params
        
    elif model_type == "gpt2":
        from model.music_gpt2 import MusicGPT2

        model = MusicGPT2(
            dset.vocab_size, model_conf['n_layer'], model_conf['n_head'],
            model_conf['d_model'], model_conf['d_ff'], model_conf['d_embed'],
            use_segment_emb=model_conf['use_segemb'], n_segment_types=model_conf['n_segment_types']
        )
        if gpuid != 'cpu': model = model.cuda(gpuid)
        
        # === 重要: パラメータ調整 (伴奏を安定させる) ===
        temp, top_p = 1.0, 0.9

    else:
        raise NotImplementedError("Unsuppported model:", model_type)
    
    print(f"[info] temp = {temp} | top_p = {top_p}")

    pretrained_dict = torch.load(inference_param_path, map_location='cpu')
    pretrained_dict = {
        k: v for k, v in pretrained_dict.items() if 'feature_map.omega' not in k
    }
    model_state_dict = model.state_dict()
    model_state_dict.update(pretrained_dict)
    model.load_state_dict(model_state_dict)

    model.eval()
    print('[info] model loaded')

    shutil.copy(train_conf_path, os.path.join(gen_leadsheet_dir, 'config_full.yaml'))

    if representation == 'functional':
        files = [os.path.join(gen_leadsheet_dir, i) for i in os.listdir(gen_leadsheet_dir) if 'roman.txt' in i]
    elif representation in ['remi', 'key']:
        files = [os.path.join(gen_leadsheet_dir, i) for i in os.listdir(gen_leadsheet_dir) if '.txt' in i]

    for file in files:
        out_name = '_'.join(file.split('/')[-1].split('_')[:2])
        print(file)
        
        # === 修正箇所: ファイル名からの感情推定（Q1~Q4対応版） ===
        if 'Q1' in file:
            e = 'Q1'
            emotion_token = 'Emotion_Q1'
        elif 'Q2' in file:
            e = 'Q2'
            emotion_token = 'Emotion_Q2'
        elif 'Q3' in file:
            e = 'Q3'
            emotion_token = 'Emotion_Q3'
        elif 'Q4' in file:
            e = 'Q4'
            emotion_token = 'Emotion_Q4'
        elif 'Positive' in file:
            e = 'Positive'
            emotion_token = 'Emotion_Positive'
        elif 'Negative' in file:
            e = 'Negative'
            emotion_token = 'Emotion_Negative'
        else:
            # マッチしない場合のフォールバック（例: 感情指定がない場合など）
            print(f"[Warn] Could not determine emotion for {file}, defaulting to Q1/Positive.")
            e = 'Positive'
            emotion_token = 'Emotion_Positive'

        print(e)
        
        # VocabにあるEmotionトークンを探す
        if emotion_token in dset.event2idx:
            emotion_id = dset.event2idx[emotion_token]
        else:
            # 辞書にQ1~Q4がない場合（古い辞書など）のフォールバック
            # Q1/Q4 -> Positive, Q2/Q3 -> Negative
            if e in ['Q1', 'Q4']:
                fallback = 'Emotion_Positive'
            else:
                fallback = 'Emotion_Negative'
            
            print(f"[Warn] {emotion_token} not in vocab, using {fallback}")
            emotion_id = dset.event2idx[fallback]

        tempo = dset.event2idx['Tempo_{}'.format(110)]
        
        # イベント読み込み（ここでマージ処理が走る）
        key, lead_sheet_events = read_generated_events(file, dset.event2idx)
        
        # キーのトークン取得
        key_token_str = key.strip()
        if key_token_str in dset.event2idx:
             key_id = dset.event2idx[key_token_str]
        else:
             print(f"[Warn] Unknown key {key_token_str}, fallback to Key_C")
             key_id = dset.event2idx['Key_C']
             key = 'Key_C'

        if representation in ['functional', 'key']:
            primer = [emotion_id, key_id, tempo]
        elif representation == 'remi':
            primer = [emotion_id, tempo]

        with torch.no_grad():
            generated = generate_conditional(model, dset.event2idx, dset.idx2event,
                                             lead_sheet_events, primer=primer,
                                             max_bars=max_bars, temp=temp, top_p=top_p,
                                             inadmissibles=None, model_type=model_type)

        generated = word2event(generated, dset.idx2event)
        generated = extract_midi_events_from_generation(key, generated, relative_melody=relative_melody)

        output_midi_path = os.path.join(gen_leadsheet_dir, out_name + '_' + e + '_full.mid')
        event_to_midi(
            key,
            list(chain(*generated[:max_bars])),
            mode='full',
            output_midi_path=output_midi_path
        )

        if play_midi:
            output_wav_path = os.path.join(gen_leadsheet_dir, out_name + '_' + e + '_full.wav')
            midi_to_wav(output_midi_path, output_wav_path)