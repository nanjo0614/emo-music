import os
import glob
import shutil
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_root",
        default="../EMO_Harmonizer/generation/emopia_functional_rule",
        help="EMO_Harmonizer の sample_* ディレクトリが並んでいるパス",
    )
    parser.add_argument(
        "--dst_root",
        default="generation/emopia_functional_harmonizer_two",
        help="Stage2 に食わせる roman.txt を置く先",
    )
    parser.add_argument(
        "--emotion",
        choices=["Positive", "Negative", "Both"],
        default="Positive",
        help="どの感情ラベルのリードシートを使うか",
    )
    args = parser.parse_args()

    os.makedirs(args.dst_root, exist_ok=True)

    if args.emotion == "Both":
        emo_list = ["Positive", "Negative"]
    else:
        emo_list = [args.emotion]

    sample_dirs = sorted(glob.glob(os.path.join(args.src_root, "sample_*")))
    if not sample_dirs:
        raise RuntimeError(f"src_root に sample_* が見つからない: {args.src_root}")

    idx = 0
    for sd in sample_dirs:
        for emo in emo_list:
            pattern = os.path.join(sd, f"lead_sheet_{emo}_*_roman.txt")
            files = sorted(glob.glob(pattern))
            if not files:
                # この sample には指定された感情の Roman 版が無い
                continue

            src = files[0]  # ひとまず 0 番目だけ使う
            base = f"harm_{idx:02d}_{emo}"  # 例: harm_00_Positive
            dst = os.path.join(args.dst_root, base + "_roman.txt")

            shutil.copy(src, dst)
            print(f"{src} -> {dst}")
            idx += 1

    print(f"合計 {idx} 個の roman.txt を作成しました。")


if __name__ == "__main__":
    main()
