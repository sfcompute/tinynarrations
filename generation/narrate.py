'''
python3 -m torch.distributed.run --nproc_per_node 8 --nnodes 1 narrate.py
'''
# %% imports.
import torch, torchaudio
import torch.nn.functional as F
import torch.distributed as dist
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

import os, contextlib
import numpy as np
from time import time
# %% init some globals.
use_flash = True
compile_model = True
autocast = True

torch.backends.cuda.enable_flash_sdp(use_flash)

dist.init_process_group('nccl')
cuID = int(os.environ['LOCAL_RANK'])
rank = dist.get_rank()
world_size = dist.get_world_size()
print(f'got local rank {cuID}, rank {rank}, and world size {world_size}')
torch.cuda.set_device(cuID)

def dprint(*args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)

dprint(f'flash ({use_flash}), compile_model ({compile_model}), autocast ({autocast})')

dataset_folder = './dataset/'
audio_out_folder = '' # YOUR AUDIO OUT DIRECTORY HERE. EXPECT ~14TB
tts_dir = '' # YOUR TTS DIR
model_name = 'tts_models--multilingual--multi-dataset--xtts_v2/'

out_sample_rate = 24000
threshold = torch.tensor(0.05, device='cuda')
splits = ['val', 'train']
dataset = False
# %% custom batch inference.
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
Xtts.batch_inference = batch_inference
# %% load model.
dprint('Loading model...')
config = XttsConfig()
config.load_json(tts_dir + model_name + 'config.json')
base_model = Xtts.init_from_config(config)
base_model.load_checkpoint(config, checkpoint_dir=tts_dir+model_name)
base_model.cuda()

model = torch.compile(base_model, mode='max-autotune') if compile_model else base_model
# %% cache latents.
dprint('Computing speaker latents...')
gpt_cond_latent, speaker_embedding = [l.cuda() for l in model.get_conditioning_latents(audio_path=[dataset_folder + 'alex.mp3'])]
dprint(f'latents computed, gpt conditioning latent shape {gpt_cond_latent.shape}, speaker embedding shape {speaker_embedding.shape}')
# %% dataset loading.
def get_tokenized_dataset(split):
    tokenized_file = dataset_folder + split + '_tokenized.pt'
    tokenized = torch.load(tokenized_file)
    dprint(f'loaded {split} tokenized, {len(tokenized)} items, first shape {tokenized[0].shape}')
    return tokenized
# %% pad batch utility.
def pad_batch(batch, clip_len):
    max_len = max([len(x) for x in batch])
    padded_batch = []
    for item in batch:
        padded_sample = F.pad(item, (1, max_len - len(item)), value=model.gpt.stop_text_token)
        padded_sample[0] = model.gpt.start_text_token
        padded_batch.append(padded_sample)
    padded_batch = torch.stack(padded_batch)[:, :clip_len].cuda()
    return padded_batch
# %% batch size calc.
def get_largest_batch(tokenized, test_sizes=[2,4,8,16,32], tests=2, util_ratio=0.7):
    clip_len = np.percentile([x.size(0) for x in tokenized[:int(1e6)]], 99.9)
    mems, numels, mels = [], [], []
    for _ in range(tests):
        for batch_size in test_sizes:
            torch.cuda.reset_peak_memory_stats()
            padded_batch = pad_batch(tokenized[:batch_size], -1)
            out = model.batch_inference(
                padded_batch,
                gpt_cond_latent.repeat(batch_size, 1, 1),
                speaker_embedding,
            )
            mems.append(torch.cuda.max_memory_allocated())
            numels.append(padded_batch.numel())
            mels.append(out['gpt_codes'].shape[1])

    available_mem = torch.cuda.get_device_properties(cuID).total_memory
    
    pred_elems = np.poly1d(np.polyfit(mems, numels, 1))(available_mem * util_ratio)
    pred_elems *= (np.mean(mels) / model.gpt.max_gen_mel_tokens)

    batch_size = pred_elems / clip_len

    batch_size = torch.tensor(batch_size, device=cuID)
    clip_len = torch.tensor(clip_len, device=cuID)
    dist.all_reduce(batch_size, op=dist.ReduceOp.SUM)
    dist.all_reduce(clip_len, op=dist.ReduceOp.SUM)
    batch_size = int((batch_size / world_size).cpu().numpy().item())
    clip_len = int((clip_len / world_size).cpu().numpy().item())



    dprint(f'calculated optimal batch size {batch_size}, clip len {clip_len}. Running for approx {len(tokenized) // batch_size} total itterations per gpu.')
    return batch_size, clip_len
# %%
for split in splits:
    split_start_time = time()
    os.makedirs(audio_out_folder + split + '_data', exist_ok=True)
    tokenized = get_tokenized_dataset(split)
    portion_size = len(tokenized) // world_size
    tokenized = tokenized[portion_size * (cuID):portion_size * (cuID + 1)]

    batch_size, clip_len = get_largest_batch(tokenized)
    batch_tiled_cond = gpt_cond_latent.repeat(batch_size, 1, 1)

    for i in range(0, int(len(tokenized) / batch_size)):
        batch = tokenized[i*batch_size:(i+1)*batch_size]
        padded_batch = pad_batch(batch, clip_len)

        out = model.batch_inference(
            padded_batch,
            batch_tiled_cond,
            speaker_embedding,
        )

        trimmed_wavs = []
        for wav in out['wav']:
            indices = torch.where(wav > threshold, torch.arange(wav.shape[0], device='cuda'), torch.tensor([0], device='cuda'))
            trimmed_wavs.append(wav[:torch.max(indices)])
        concatenated_wav = torch.cat(trimmed_wavs, dim=0).unsqueeze(0)

        torchaudio.save(f'{audio_out_folder + split}_data/{cuID}_{str(i)}.wav', concatenated_wav.cpu(), 24000)

    print(f'GPU {cuID} finished {split} split in {time() - split_start_time} seconds.')
    dist.barrier()