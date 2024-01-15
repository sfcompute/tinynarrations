# Tiny Narrations
> 3.5 years worth of synthetically narrated children's stories. Scripts written by GPT4 from [TinyStories](https://arxiv.org/abs/2305.07759).
[Blogpost.](https://sfcompute.com/tiny-narrations)


### Instructions.
`git clone https://github.com/sfcompute/tinynarrations.git`
`cd ./tinynarrations`

`pip install boto3` and login. Transfer to other AWS instances is free, otherwise expensive egress for the larger splits.


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

[sample](https://sfcompute.com/media/tinynarrations.webm)


### Generation
As of now we don't have standardized scripts for generation of similar datasets. The main bit is just a batch inference function. To run batch inference on XTTS-v2, we used the following modified class method and the original TTS library:
```python
def batch_inference(self, text_tokens, gpt_cond_latent, speaker_embedding,
        temperature=0.75, length_penalty=1.0, repetition_penalty=10.0, top_k=50, top_p=0.85, do_sample=True, num_beams=1, speed=1.0, **hf_generate_kwargs,
    ):
        wavs = []
        gpt_latents_list = []

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16) if autocast else contextlib.nullcontext():
                gpt_codes = self.gpt.generate(
                    cond_latents=gpt_cond_latent,
                    text_inputs=text_tokens,
                    input_tokens=None,
                    do_sample=do_sample,
                    # etc. (gpt_batch_size is 1, we're hacking around it.)
                    top_p=top_p, top_k=top_k, temperature=temperature, num_return_sequences=self.gpt_batch_size, num_beams=num_beams, length_penalty=length_penalty, repetition_penalty=repetition_penalty, output_attentions=False, **hf_generate_kwargs,
                )
                expected_output_len = torch.tensor(
                    [gpt_codes.shape[-1] * self.gpt.code_stride_len], device=text_tokens.device
                )

                text_len = torch.tensor([text_tokens.shape[-1]], device=self.device)
                gpt_latents = self.gpt(
                    text_tokens,
                    text_len,
                    gpt_codes,
                    expected_output_len,
                    cond_latents=gpt_cond_latent,
                    return_attentions=False,
                    return_latent=True,
                )

                if speed != 1.0:
                    gpt_latents = F.interpolate(
                        gpt_latents.transpose(1, 2), scale_factor=(1.0 / max(speed, 0.05)), mode='linear'
                    ).transpose(1, 2)

                wav = self.hifigan_decoder(gpt_latents, g=speaker_embedding)

            gpt_latents_list.append(gpt_latents.cpu().float())
            wavs.append(wav.squeeze().float())

        return {
            'wav': torch.cat(wavs, dim=0),
            'gpt_latents': torch.cat(gpt_latents_list, dim=1).numpy(),
            'gpt_codes': gpt_codes.cpu().numpy(),
            'speaker_embedding': speaker_embedding,
        }
```
