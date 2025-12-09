import os

# 1. 結合したい拡張子（Pythonコード、ドキュメント、設定ファイル）
TARGET_EXTENSIONS = ['.py', '.md', '.txt', '.yaml', '.json', '.sh'] 

# 2. 無視したいフォルダ（モデルの重み、バックアップ、キャッシュ、Gitなど）
IGNORE_DIRS = {
    'best_weight',       # モデルの重みファイル（バイナリ）
    'emo-music-backup',  # バックアップ（重複回避）
    'experiment_temp',   # 一時的な出力結果
    '__pycache__',       # Pythonキャッシュ
    '.git',              # Git管理用
    '.idea', '.vscode',  # エディタ設定
    'wandb',             # 実験管理ログ（あれば）
    'venv', 'env'        # 仮想環境
}

# 3. 無視したい特定のファイル（もしあれば）
IGNORE_FILES = {
    'tree_output.txt',   # 構成図自体はコードではないので除外してもOK
    '.DS_Store'          # Macのシステムファイル
}

# 出力ファイル名
OUTPUT_FILE = 'project_full_context.txt'

def merge_files():
    root_dir = os.getcwd()
    print(f"Scanning directory: {root_dir} ...")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        # 冒頭にツリー構造（構成図）を入れておくとAIが全体像を把握しやすくなります
        if os.path.exists('tree_output.txt'):
             with open('tree_output.txt', 'r', encoding='utf-8') as tree_f:
                 outfile.write("=== PROJECT DIRECTORY STRUCTURE ===\n")
                 outfile.write(tree_f.read())
                 outfile.write("\n======================================\n\n")

        for dirpath, dirnames, filenames in os.walk(root_dir):
            # 無視リストにあるフォルダを探索対象から外す（その下のファイルも見に行かない）
            dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
            
            for filename in filenames:
                # 拡張子チェック
                if any(filename.endswith(ext) for ext in TARGET_EXTENSIONS):
                    # 無視ファイルリストに含まれていないか確認
                    if filename in IGNORE_FILES or filename == os.path.basename(__file__) or filename == OUTPUT_FILE:
                        continue

                    file_path = os.path.join(dirpath, filename)
                    relative_path = os.path.relpath(file_path, root_dir)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            content = infile.read()
                            
                            # AIへの区切り線
                            outfile.write(f"\n{'='*60}\n")
                            outfile.write(f"File: {relative_path}\n")
                            outfile.write(f"{'='*60}\n")
                            outfile.write(content + "\n")
                            print(f"Merged: {relative_path}")
                    except Exception as e:
                        print(f"Skipped {relative_path}: {e}")

    print(f"\nDone! Output file: '{OUTPUT_FILE}' created.")

if __name__ == '__main__':
    merge_files()