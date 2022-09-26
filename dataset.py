from torch.utils.data import Dataset
import torch
from tqdm import tqdm
from data_process.data_loader import read_row_data
from data_process.preprocessor import add_prompt


class DevDataset(Dataset):
    def __init__(self, dev_data, tokenizer):
        self.tokenizer = tokenizer
        self.support_encoded_prompt_in = []
        self.support_encoded_prompt_out = []
        self.support_encoded_raw_sent = []
        self.support_mask_encoded_prompt_out = []

        self.support_encoded_checker_prompt_in = []
        self.support_encoded_checker_prompt_out = []
        self.support_encoded_checker_raw_sent = []
        self.support_mask_encoded_checker_prompt_out = []

        self.original_seq_in = []
        self.original_bio_seq_out = []


        for example in tqdm(dev_data, desc='Load dev data', ncols=80):
            support_prompt_in = example['prompt_seq_in']
            support_prompt_out = example['prompt_seq_out']
            support_checker_prompt_in = example['checker_prompt_in']
            support_checker_prompt_out = example['checker_prompt_out']
            
            encoded_support_prompt_in = [tokenizer(" ".join(sent_label).replace("'", '"'), add_prefix_space=True)['input_ids'] for sent_label in support_prompt_in]
            encoded_support_prompt_out = [tokenizer(" ".join(sent_label).replace("'", '"') + " .<|endoftext|>", add_prefix_space=True)['input_ids'] for sent_label in support_prompt_out]
            encoded_support_checker_prompt_in = [tokenizer(" ".join(sent_label).replace("'", '"'), add_prefix_space=True)['input_ids'] for sent_label in support_checker_prompt_in]
            encoded_support_checker_prompt_out = [tokenizer(" ".join(sent_label).replace("'", '"') + " .<|endoftext|>",add_prefix_space=True)['input_ids'] for sent_label in support_checker_prompt_out]
            
            
            
            encoded_support_mask_prompt_out = [len(sent) * [-100] + encoded_support_prompt_out[sid][len(sent): ] for sid, sent in enumerate(encoded_support_prompt_in)]
            encoded_support_mask_checker_prompt_out = [len(sent) * [-100] + encoded_support_checker_prompt_out[sid][len(sent): ] for sid,sent in enumerate(encoded_support_checker_prompt_in)]
            
            self.support_encoded_prompt_in.append(encoded_support_prompt_in)
            self.support_encoded_prompt_out.append(encoded_support_prompt_out)
            self.support_encoded_checker_prompt_in.append(encoded_support_checker_prompt_in)
            self.support_encoded_checker_prompt_out.append(encoded_support_checker_prompt_out)
            self.support_mask_encoded_prompt_out.append(encoded_support_mask_prompt_out)
            self.support_mask_encoded_checker_prompt_out.append(encoded_support_mask_checker_prompt_out)



            onesent_encoded_raw_sent = []
            onesent_encoded_checker_raw_sent = []


            raw_sent = example['original_seq_in']
            encoded_raw_sent = tokenizer.encode(" ".join(raw_sent), add_prefix_space=True)
            encoded_raw_sent.append(2162)
            encoded_raw_sent.append(4844)
            encoded_raw_sent.append(764)
            encoded_raw_sent.append(tokenizer.eos_token_id)
            for _ in range(len(example['prompt_seq_in'])):
                onesent_encoded_raw_sent.append(encoded_raw_sent)

            raw_sent = example['original_seq_in']
            encoded_raw_sent = tokenizer.encode(" ".join(raw_sent), add_prefix_space=True)
            encoded_raw_sent.append(2162)
            encoded_raw_sent.append(4844)
            encoded_raw_sent.append(764)
            encoded_raw_sent.append(tokenizer.eos_token_id)
            for _ in range(len(example['prompt_seq_in'])):
                onesent_encoded_checker_raw_sent.append(encoded_raw_sent)


            self.support_encoded_raw_sent.append(onesent_encoded_raw_sent)
            self.support_encoded_checker_raw_sent.append(onesent_encoded_checker_raw_sent)

            self.original_seq_in.append(example['original_seq_in'])
            self.original_bio_seq_out.append(example['original_bio_seq_out'])


    def __len__(self):
        return len(self.support_encoded_prompt_in)

    def __getitem__(self, index):
        return {
            'prompt_in': self.support_encoded_prompt_in[index],
            'prompt_out': self.support_encoded_prompt_out[index],
            'mask_prompt_out': self.support_mask_encoded_prompt_out[index],
            'checker_prompt_in': self.support_encoded_checker_prompt_in[index],
            'checker_prompt_out': self.support_encoded_checker_prompt_out[index],
            'raw_sent': self.support_encoded_raw_sent[index],
            'checker_raw_sent': self.support_encoded_checker_raw_sent[index],
            'mask_checker_prompt_out': self.support_mask_encoded_checker_prompt_out[index],
            'original_seq_in': self.original_seq_in[index],
            'original_bio_seq_out': self.original_bio_seq_out[index]
        }



class TrainDataset(Dataset):
    def __init__(self, dev_data, tokenizer):
        self.tokenizer = tokenizer
        self.support_encoded_prompt_in = []
        self.support_encoded_prompt_out = []
        self.support_mask_encoded_prompt_out = []



        for example in tqdm(dev_data, desc='Load train data', ncols=80):
            support_prompt_in = example['prompt_seq_in']
            support_prompt_out = example['prompt_seq_out']
            
            encoded_support_prompt_in = [tokenizer(" ".join(sent_label).replace("'", '"'), add_prefix_space=True)['input_ids'] for sent_label in support_prompt_in]
            encoded_support_prompt_out = [tokenizer(" ".join(sent_label).replace("'", '"') + " .<|endoftext|>", add_prefix_space=True)['input_ids'] for sent_label in support_prompt_out]

            encoded_support_mask_prompt_out = [len(sent) * [-100] + encoded_support_prompt_out[sid][len(sent): ] for sid, sent in enumerate(encoded_support_prompt_in)]
            
            self.support_encoded_prompt_in.extend(encoded_support_prompt_in)
            self.support_encoded_prompt_out.extend(encoded_support_prompt_out)
            self.support_mask_encoded_prompt_out.extend(encoded_support_mask_prompt_out)




    def __len__(self):
        return len(self.support_encoded_prompt_in)



    def __getitem__(self, index):
        return {
            'promp_in': self.support_encoded_prompt_in[index],
            'prompt_out': self.support_encoded_prompt_out[index],
            'mask_prompt_out': self.support_mask_encoded_prompt_out[index]
        }



class CheckerTrainDataset(Dataset):
    def __init__(self, dev_data, tokenizer):
        self.tokenizer = tokenizer
        self.support_encoded_checker_prompt_in = []
        self.support_encoded_checker_prompt_out = []
        self.support_encoded_checker_raw_sent = []
        self.support_mask_encoded_checker_prompt_out = []


        for example in tqdm(dev_data, desc='Load train data', ncols=80):
            support_checker_prompt_in = example['checker_prompt_in']
            support_checker_prompt_out = example['checker_prompt_out']
            
            encoded_support_checker_prompt_in = [tokenizer(" ".join(sent_label).replace("'", '"'), add_prefix_space=True)['input_ids'] for sent_label in support_checker_prompt_in]
            encoded_support_checker_prompt_out = [tokenizer(" ".join(sent_label).replace("'", '"') + " .<|endoftext|>",add_prefix_space=True)['input_ids'] for sent_label in support_checker_prompt_out]
            
            
            
            encoded_support_mask_checker_prompt_out = [len(sent) * [-100] + encoded_support_checker_prompt_out[sid][len(sent): ] for sid,sent in enumerate(encoded_support_checker_prompt_in)]
            
            self.support_encoded_checker_prompt_in.extend(encoded_support_checker_prompt_in)
            self.support_encoded_checker_prompt_out.extend(encoded_support_checker_prompt_out)
            self.support_mask_encoded_checker_prompt_out.extend(encoded_support_mask_checker_prompt_out)

            onesent_encoded_checker_raw_sent = []


            raw_sent = example['original_seq_in']
            encoded_raw_sent = tokenizer.encode(" ".join(raw_sent), add_prefix_space=True)
            encoded_raw_sent.append(2162)
            encoded_raw_sent.append(4844)
            encoded_raw_sent.append(764)
            encoded_raw_sent.append(tokenizer.eos_token_id)
            for _ in range(len(example['prompt_seq_in'])):
                onesent_encoded_checker_raw_sent.append(encoded_raw_sent)

            self.support_encoded_checker_raw_sent.extend(onesent_encoded_checker_raw_sent)


    def __len__(self):
        return len(self.support_encoded_prompt_in)


    def __getitem__(self, index):
        return {
            'promp_in': self.support_encoded_checker_prompt_in[index],
            'prompt_out': self.support_encoded_checker_prompt_out[index],
            'mask_prompt_out': self.support_mask_encoded_checker_prompt_out[index]
        }



def train_collate_fn(batch):
    output = {}
    max_length = max([len(x['prompt_out']) for x in batch])
    attention_mask = [[1] * len(x['prompt_out']) + [0] * (max_length - len(x['prompt_out'])) for x in batch]
    prompt_out = [x['prompt_out'] + [50256] * (max_length - len(x['prompt_out'])) for x in batch]
    mask_prompt_out = [x['mask_prompt_out'] + [-100] * (max_length - len(x['mask_prompt_out'])) for x in batch]
    output['prompt_out'] = torch.tensor(prompt_out, dtype=torch.long)
    output['attention_mask'] = torch.tensor(attention_mask, dtype=torch.long)
    output['mask_prompt_out'] = torch.tensor(mask_prompt_out, dtype=torch.long)
    return output





class GPT2PromptDevDataset(Dataset):
    def __init__(self, support_encoded_prompt_in, support_encoded_raw_sent):
        self.support_encoded_prompt_in = support_encoded_prompt_in
        self.support_encoded_raw_sent = support_encoded_raw_sent

    def __len__(self):
        return len(self.support_encoded_prompt_in)

    def __getitem__(self, index):
        return {
            'prompt_in': self.support_encoded_prompt_in[index],
            'encoded_raw_sent': self.support_encoded_raw_sent[index]
        }


def dev_collate_fn(batch):
    output = {}
    max_length = max([len(x['prompt_in']) for x in batch])
    batch_length = [len(x['prompt_in']) for x in batch]
    prompt_in = [x['prompt_in'] + [50256] * (max_length - len(x['prompt_in'])) for x in batch]
    max_raw_length = max([len(x['encoded_raw_sent']) for x in batch])
    encoded_raw_sent = [x['encoded_raw_sent'] + [50256] * (max_raw_length - len(x['encoded_raw_sent'])) for x in batch]
    output['prompt_in'] = torch.tensor(prompt_in, dtype=torch.long)
    output['batch_length'] = batch_length
    output['encoded_raw_sent'] = torch.tensor(encoded_raw_sent, dtype=torch.long)

    return output



def get_dataset(mode, domain, nsamples, tokenizer, is_checker=False):
    train_set, dev_set, test_set = read_row_data(domain, nsamples)
    train_set = add_prompt(train_set)
    dev_set = add_prompt(dev_set)
    test_set = add_prompt(test_set)
    if mode == 'train':
        if is_checker:
            dataset = CheckerTrainDataset(train_set, tokenizer)
        else:
            dataset = TrainDataset(train_set, tokenizer)
    elif mode == 'dev':
        dataset = DevDataset(dev_set, tokenizer)
    elif mode == 'test':
        dataset = DevDataset(test_set, tokenizer)
    return dataset



if __name__ == '__main__':
    from transformers import GPT2Tokenizer
    from data_process.data_loader import read_row_data
    from data_process.preprocessor import add_prompt
    from torch.utils.data import DataLoader
    train_set, dev_set, _ = read_row_data('AddToPlaylist', 20)
    train_data = add_prompt(train_set)
    dev_data = add_prompt(dev_set)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    dev_dataset = DevDataset(dev_data, tokenizer)
    print(len(dev_dataset))
    for data in dev_dataset:
        tmp_dataset = GPT2PromptDevDataset(data['prompt_in'], data['raw_sent'])
        tmp_dataloader = DataLoader(tmp_dataset, batch_size=2, shuffle=False, collate_fn=dev_collate_fn)
        print(len(tmp_dataloader))
        for i in tmp_dataloader:
            print(i)
        break
        