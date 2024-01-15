# Tiny Narrations
> 3 and a half years of synthetically narrated children's stories written by GPT4. Based on [TinyStories](https://arxiv.org/abs/2305.07759)

Sample code to synthesize the dataset on an h100 node is included in the `/generations` folder. Audio download script included at `download.py`.

Modify the download script to get the data you're interested in. There are two main raw audio folders, `/train_data` and `/val_data`, which contain synthetically generated wav files from xttsV2. Batch inference scripts are included in the generations folder. For smaller downloads, pretokenized data is available, with [Hubert](https://github.com/facebookresearch/fairseq/blob/main/examples/hubert/README.md) for semantic tokens and [Encodec](https://github.com/facebookresearch/encodec) for decodable audio tokens, assuming an AudioLM style approach.

[sample](https://sfcompute.com/media/tinynarrations.webm)

