from data_process.constants import domain2slots, label_verb
from data_process.data_loader import read_row_data
import random

labelmap = label_verb


def add_prompt(examples):
    processed_examples = []
    for example in examples:
        seq_in = example['seq_in']
        seq_out = example['seq_out']
        bio_seq_out = example['bio_seq_out']
        domain_name = example['domain']


        prompt_ins = []
        prompt_outs = []
        checker_prompt_ins = []
        checker_prompt_outs= []

        # sample check pair
        label_entity_pair = {}  # slot label对应的slot entity

        for token, label, bio_label in zip(seq_in, seq_out, bio_seq_out):
            if bio_label.startswith('B-'):
                if label not in label_entity_pair:
                    label_entity_pair[label]=[token]
                else:
                    label_entity_pair[label].append(token)
            if bio_label.startswith('I-'):
                label_entity_pair[label][-1] += ' '+token
        
        all = list(label_entity_pair.items())
        if len(all) != 0:
            unsampled = random.sample(all, 1)  # 随机采样一个slot_entity做interative
        else:
            unsampled = []
        sampled = [i for i in all if i not in unsampled]
        all = dict(all)
        sampled = dict(sampled)
        unsampled = dict(unsampled)

        

        checker_prompt_in =  ["'"] + seq_in + ["'"]
        checker_prompt_out = ["'"] + seq_in + ["'"]
        for label in domain2slots[domain_name]:
            if label in sampled:
                mapped_label = labelmap[domain_name][label]
                checker_prompt_in += mapped_label.split() + ["refers"] + ["to"]
                checker_prompt_out += mapped_label.split() + ["refers"] + ["to"]
                for slot_value in label_entity_pair[label]:
                    checker_prompt_in += slot_value.split() + [";"]
                    checker_prompt_out += slot_value.split() + [";"]
                    
                if checker_prompt_in[-1] == ';':
                    checker_prompt_in = checker_prompt_in[:-1]
                    checker_prompt_out = checker_prompt_out[:-1]
                    checker_prompt_in += ["."]
                    checker_prompt_out += ["."]
        
        for label in domain2slots[domain_name]:
            one_checker_prompt_in = [i for i in checker_prompt_in]
            one_checker_prompt_out = [i for i in checker_prompt_out]
            if label in unsampled:
                mapped_label = labelmap[domain_name][label] 
                one_checker_prompt_in += mapped_label.split() + ["refers"] + ["to"] 
                one_checker_prompt_out += mapped_label.split() + ["refers"] + ["to"]
                for slot_value in label_entity_pair[label]:
                    one_checker_prompt_out += slot_value.split() + [";"]
                if one_checker_prompt_out[-1]==';':
                    one_checker_prompt_out = one_checker_prompt_out[: -1]
                
                checker_prompt_ins.append(one_checker_prompt_in)
                checker_prompt_outs.append(one_checker_prompt_out)
            if label not in all:
                mapped_label = labelmap[domain_name][label] 
                one_checker_prompt_in += mapped_label.split() + ["refers"] + ["to"] 
                one_checker_prompt_out += mapped_label.split() + ["refers"] + ["to"] + ['none']
                checker_prompt_ins.append(one_checker_prompt_in)
                checker_prompt_outs.append(one_checker_prompt_out)
        
        # construct first round prompt
        for label in domain2slots[domain_name]:
            prompt_in =  ["'"] + seq_in + ["'"]
            prompt_out = ["'"] + seq_in + ["'"]
            if label in seq_out:
                mapped_label = labelmap[domain_name][label]
                prompt_in +=  mapped_label.split() + ["refers"] + ["to"]
                prompt_out += mapped_label.split() + ["refers"] + ["to"]
                label_mask = [1 if l == label else 0 for l in seq_out]

                for idx, mask in enumerate(label_mask):
                    if mask == 1:
                        if bio_seq_out[idx].startswith('B-'):
                            label_mask[idx] = 1
                        if bio_seq_out[idx].startswith('I-'):
                            label_mask[idx] = -1
                for idx, mask in enumerate(label_mask):
                    if mask == 1:
                        prompt_out = prompt_out + [seq_in[idx]]
                    if mask == -1:
                        prompt_out = prompt_out + [seq_in[idx]]
                    if idx >= 1:
                        if label_mask[idx - 1] == -1 and mask == 0:
                            prompt_out = prompt_out + [';']
                if prompt_out[-1] == ';':
                    prompt_out = prompt_out[:-1]
            else:
                mapped_label = labelmap[domain_name][label]
                prompt_in += mapped_label.split() + ["refers"] + ["to"]
                prompt_out +=  mapped_label.split() + ["refers"] + ["to"] + ["none"]
            prompt_outs.append(prompt_out)
            prompt_ins.append(prompt_in)
        
        processed_examples.append({
            'domain': domain_name,
            'prompt_seq_in': prompt_ins,
            'prompt_seq_out': prompt_outs,
            'checker_prompt_in': checker_prompt_ins,
            'checker_prompt_out': checker_prompt_outs,
            'original_seq_in': example['seq_in'],
            'original_seq_out': example['seq_out'],
            'original_bio_seq_out': example['bio_seq_out']
        })
    
    return processed_examples



if __name__ == "__main__":
    train_set, _, _ = read_row_data('AddToPlaylist', 0)
    train_data = add_prompt(train_set)
    print(len(train_data[1]['prompt_seq_in']))
    print(len(train_data[1]['prompt_seq_out']))
    print(len(train_data[1]['checker_prompt_in']))
    print(len(train_data[1]['checker_prompt_out']))