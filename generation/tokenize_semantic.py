'''
python3 -m torch.distributed.run --nproc_per_node 8 --nnodes 1 tokenize_semantic.py
'''

import os
from torch.utils.data import Dataset, DataLoader
import torch
import torchaudio
import torch.distributed as dist
from audiolm_pytorch import HubertWithKmeans

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

dist.init_process_group('nccl')
cuID = int(os.environ['LOCAL_RANK'])
rank = dist.get_rank()
world_size = dist.get_world_size()
print(f'got local rank {cuID}, rank {rank}, and world size {world_size}')
torch.cuda.set_device(cuID)


class WavsData(Dataset):
    def __init__(self, wavs_dir):
        self.wavs_dir = wavs_dir

    def __len__(self):
        file_count = 0
        for filename in os.listdir(self.wavs_dir):
            if filename.startswith(str(cuID)):
                file_count += 1
        return file_count

    def __getitem__(self, idx):
        wav_path = os.path.join(self.wavs_dir, f'{str(cuID)}_{str(idx)}.wav')
        wav, _ = torchaudio.load(wav_path)
        wav = wav.flatten()
        wav = torch.as_strided(wav, (wav.shape[0] // (24000*30-240) - 1, 24000*30), (24000*30-240, 1))
        return wav.to(torch.bfloat16)

class WrapperCast(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):   
        batch = batch.squeeze()
        semantic_tokens = self.model(
            batch,
            flatten = False,
            input_sample_hz = 24000
        )
        return semantic_tokens


uncompiled_model = HubertWithKmeans(
    checkpoint_path='./hubert/hubert_base_ls960.pt',
    kmeans_path='./hubert/hubert_base_ls960_L9_km500.bin'
).cuda()

wrapped_model = WrapperCast(uncompiled_model)
# model = torch.compile(wrapped_model, mode='max-autotune-no-cudagraphs', fullgraph=False)
model = wrapped_model


for split in ['val', 'train']:
    print(split, cuID)

    wavydata = WavsData(f'/YOUR_PATH_HERE/{split}_data')

    dataloader = DataLoader(wavydata, batch_size=1, prefetch_factor=2, num_workers=4, pin_memory=False)

    for i, batch in enumerate(dataloader):
        if i == 2:
            break
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            audio_tokens = model(batch.cuda())

        torch.save(audio_tokens.cpu().detach(), f'/YOUR_PATH_HERE/{split}_data_semantic_tokenized/{str(cuID)}_{str(i)}.pt')