# Tiny Narrations
> 3.5 years worth of synthetically narrated children's stories. Scripts written by GPT4 from [TinyStories](https://arxiv.org/abs/2305.07759).

### [Release page](https://sfcompute.com/blog/tiny-narrations)
### [Huggingface dataset](https://huggingface.co/datasets/sfcompute/tiny-narrations)

Listen to a [sample](https://sfcompute.com/media/tinynarrations.webm).
<br>
<br>

## Instructions (Huggingface datasets)
```bash
pip install datasets
```

```python
from datasets import load_dataset

val_split = load_dataset('sfcompute/TinyNarrations', split='validation', streaming=True)
train_split = load_dataset('sfcompute/TinyNarrations', split='train', streaming=True)
```

```python
import torch

wav = torch.from_numpy(next(iter(val_split))['audio']['array']).unsqueeze(0)
```


To load audio ensure you have the following installed:
```bash
pip install librosa soundfile
```

### Instructions (S3 bucket)
```
git clone https://github.com/sfcompute/tinynarrations.git
cd ./tinynarrations
pip install boto3
```
Then login to AWS. Transfer to other AWS instances is free, otherwise expensive egress for the larger splits.


Edit line 20 to specify the data splits you'd like to download.
```python
"""
Folders:
/train_data - ~14TB, ~90k wav files
/val_data - ~137GB, 864 wav files

/train_data_semantic_tokenized - ~56GB, ~90k pt files
/val_data_semantic_tokenized - ~573MB, 864 pt files

/train_data_encodec_tokenized - ~687GB, ~90k pt files
/val_data_encodec_tokenized - ~7GB, 864 pt files

"""

folders = ['val_data_encodec_tokenized'] # ADD FOLDERS HERE
```

There are two main raw audio folders, `/train_data` and `/val_data`, which contain synthetically generated wav files. Batch inference scripts are included in the generations folder. For smaller downloads, pretokenized data is available (`val_data_semantic_tokenized`, `val_data_encodec_tokenized` etc.), with [Hubert](https://github.com/facebookresearch/fairseq/blob/main/examples/hubert/README.md) for semantic tokens and [Encodec](https://github.com/facebookresearch/encodec) for decodable audio tokens, assuming an AudioLM style approach.

To decode tokenized Encodec data:
`pip install -U encodec`

```python
encodec = EncodecModel.encodec_model_24khz().cuda()
encodec.set_target_bandwidth(6.0)

encoded_tokens = torch.load(path)
print(encoded_tokens.shape) # (n 1-secs, 8 quantizers, 75 tps)
frame_list = [(frame.unsqueeze(0).cuda(), None) for frame in sample_encoded_tokens]
print(frame_list[0][0].shape)
encodec.segment = 1

with torch.no_grad():
        decoded_waveform = encodec.decode(frame_list)
```