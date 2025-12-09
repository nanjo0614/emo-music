import os
from midi2audio import FluidSynth

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GEN_DIR = os.path.join(BASE_DIR, "generation", "emopia_functional_two")

# inference(stage1/2).pyと同じ想定
SOUNDFONT = os.path.join(
    BASE_DIR,
    "SalamanderGrandPiano-SF2-V3+20200602",
    "SalamanderGrandPiano-V3+20200602.sf2",
)

def main():
    if not os.path.exists(SOUNDFONT):
        raise FileNotFoundError(f"SoundFont not found: {SOUNDFONT}")

    fs = FluidSynth(SOUNDFONT)

    files = sorted(os.listdir(GEN_DIR))
    mid_files = [f for f in files if f.endswith("_full.mid") and any(q in f for q in ["_Q1_", "_Q2_", "_Q3_", "_Q4_"])]

    print(f"Found {len(mid_files)} full MIDIs")

    for fname in mid_files:
        midi_path = os.path.join(GEN_DIR, fname)
        wav_path = midi_path.replace(".mid", ".wav")

        if os.path.exists(wav_path):
            print(f"[skip] {wav_path} already exists")
            continue

        print(f"[render] {midi_path} -> {wav_path}")
        fs.midi_to_audio(midi_path, wav_path)

    print("done.")

if __name__ == "__main__":
    main()
