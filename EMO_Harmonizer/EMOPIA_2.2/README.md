
# EMOPIA

- `midis/`: midi files transcribed using GiantMIDI.
    * Filename `Q1_xxxxxxx_2.mp3`: Q1 means this clips belongs to Q1 on V-A space; xxxxxxx is the song ID on YouTube; and the `2` means this clip is the 2nd clips taken from the full song.
- `metadata/`: metadata from YouTube. (Got when crawling)
- `songs_lists/`: YouTube URLs of songs.
- `tagging_lists/`: raw tagging result for each sample.
- `label.csv`: metadata that record filename, 4Q label, and annotator.
- `timestamps.json`: timestamps for every clips. The format is dict. The key is the YouTube ID and the value is the timestamps for every clip in the song. You can see `scripts/load_timestamp.ipynb` for the format example.
- `metadata_by_song.csv`: list all the clips by the song. Can be used to create the train/val/test splits to avoid the same song appear in both train and test.
- `scripts/prepare_split.ipynb`: the script to create train/val/test splits and save them to csv files.
- `scripts/load_timestamp.ipynb`: shows the format of the timestamp.
- `scripts/timestamp2clip.py`: After the raw audio are crawled and put in `audios/raw`, you can use this script to get audio clips. The script will read `timestamps.json` and use the timestamp to extract clips. The clips will be saved to `audios/seg` folder.