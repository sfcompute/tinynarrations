import os
import re
import textwrap
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from datasets import load_dataset
from spacy.lang.en import English

dataset = load_dataset("roneneldan/TinyStoriesInstruct")
dataset_folder = './dataset/'

tts_dir = '' # YOUR TTS DIRECTORY HERE
model_name = 'tts_models--multilingual--multi-dataset--xtts_v2/'
config = XttsConfig()
config.load_json(tts_dir + model_name + 'config.json')
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=tts_dir+model_name)
model.cuda()


def split_sentence(text, split_len=250):
    text_splits = []
    if len(text) >= split_len:
        text_splits.append('')
        max_chunk = 2**30-1
        for big_chunk in range(0, len(text), max_chunk):
            text_block = text[big_chunk:big_chunk+max_chunk]
            nlp = English()
            nlp.max_length = len(text_block) + 1
            nlp.add_pipe('sentencizer')
            doc = nlp(text_block)
            for sentence in doc.sents:
                sentlen = len(str(sentence))
                if len(text_splits[-1]) + sentlen <= split_len:
                    text_splits[-1] += ' ' + str(sentence)
                    text_splits[-1] = text_splits[-1].lstrip()
                elif sentlen > split_len:
                    for line in textwrap.wrap(str(sentence), width=split_len, drop_whitespace=True, break_on_hyphens=False, tabsize=1,):
                        text_splits.append(str(line))
                else:
                    text_splits.append(str(sentence))

            if len(text_splits) > 1:
                if text_splits[0] == '':
                    del text_splits[0]
    else:
        text_splits = [text.lstrip()]
    return text_splits

def get_tokenized_dataset(split):
    tokenized_file = dataset_folder + split + '_tokenized.pt'
    text_file = dataset_folder + 'dataset_' + split + '_text.txt'
    if os.path.exists(tokenized_file):
        tokenized = torch.load(tokenized_file)
        pass
    else:
        # load text
        if os.path.exists(text_file):
            with open(text_file, 'r') as f:
                split_text = f.read()
        else:
            split_text = '\n'.join(dataset['validation']['text'] if split == 'val' else dataset['train']['text'])
            split_text = re.sub(r'(?<=\w)\n', '. \n', split_text)
            with open(text_file, 'w') as f:
                f.write(split_text)
        print(f'loaded {split} text, {len(split_text)} characters')

        # tokenizing
        model.tokenizer.check_input_length = lambda *args, **kwargs: None
        split_text = split_sentence(split_text,  model.tokenizer.char_limits['en'])
        tokenized = []
        for text in split_text:
            text = text.strip().lower()
            tokenized.append(torch.IntTensor(model.tokenizer.encode(text, lang='en')))
        torch.save(tokenized, tokenized_file)
    print(f'loaded {split} tokenized, {len(tokenized)} items, first shape {tokenized[0].shape}')

    return tokenized
print('tokenizing')
model.tokenizer.check_input_length = lambda *args, **kwargs: None
val_tokenized = get_tokenized_dataset('val')
train_tokenized = get_tokenized_dataset('train')