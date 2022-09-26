import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AdamW, AdamW, GPT2Tokenizer, get_linear_schedule_with_warmup
from model import GPT2GenModel
import logging
from dataset import get_dataset, train_collate_fn, dev_collate_fn, GPT2PromptDevDataset
from tqdm import tqdm
import os
import numpy as np
import argparse
from logging import StreamHandler, FileHandler
import sys
import random
from data_process.constants import verb_label
from seqeval.metrics import precision_score, recall_score, f1_score


logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_logger(log_filename):
    handler1 = StreamHandler(stream=sys.stdout)
    handler2 = FileHandler(filename=log_filename, mode='a', delay=False)
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[handler1, handler2]
    )


class Trainer():
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2GenModel.from_pretrained("gpt2")
        self.model.to(self.device)
        train_dataset = get_dataset('train', self.args.domain, self.args.nsamples, tokenizer, is_checker=False)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, collate_fn=train_collate_fn)
        self.dev_dataset = get_dataset('dev', self.args.domain, self.args.nsamples, tokenizer, is_checker=False)
        self.test_dataset = get_dataset('test', self.args.domain, self.args.nsamples, tokenizer, is_checker=False)

        self.tokenizer = tokenizer
        t_total = len(self.train_dataloader) * self.args.num_train_epochs
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.args.warmup_rate*t_total, num_training_steps=t_total)



    def train(self):
        early_stop_cnt = 0
        max_f1 = -1

        for epoch in range(self.args.num_train_epochs):
            train_loss = 0.0
            for _, batch in enumerate(tqdm(self.train_dataloader, desc=f'Train Epoch {epoch + 1}/{self.args.num_train_epochs}', ncols=80)):
                self.model.train()
                self.optimizer.zero_grad()

                prompt_out = batch['prompt_out'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                mask_prompt_out = batch['mask_prompt_out'].to(self.device)


                outputs = self.model(prompt_out, labels=mask_prompt_out, attention_mask=attention_mask)
                loss = outputs[0]

                with torch.no_grad():
                    train_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            with torch.no_grad():
                acc, recall, f1 = self.evaluate('dev')
            logger.info(f'Epoch {epoch + 1}, train_loss {train_loss / len(self.train_dataloader):.6f}, Evalution acc {acc:.4f}, recall {recall:.4f}, f1 {f1:.4f}')
            if f1 > max_f1:
                max_f1 = f1
                early_stop_cnt = 0
                self.save_model(epoch + 1)
            else:
                early_stop_cnt += 1
            if early_stop_cnt >= self.args.early_stop:
                logger.info(f'Reach early stop count {self.args.early_stop}, training stop')
                break


    def evaluate(self, mode):
        if mode == 'dev':
            dataset = self.dev_dataset
        if mode == 'test':
            dataset = self.test_dataset
        
        all_pred_tags = []
        all_gold_tags = []
        label_map = verb_label
        domain_name = self.args.domain

        for dev_ep in tqdm(dataset, desc='Dev' if mode == 'dev' else 'Test', ncols=80):
            prompt_in = dev_ep['prompt_in']
            raw_sent = dev_ep['raw_sent']
            original_bio_seq_out = dev_ep['original_bio_seq_out']
            original_seq_in = dev_ep['original_seq_in']
            pred_tags = ['O'] * len(original_seq_in)

            dev_dataset = GPT2PromptDevDataset(prompt_in, raw_sent)
            dev_dataloader = DataLoader(dev_dataset, batch_size=self.args.dev_batch_size, shuffle=False, collate_fn=dev_collate_fn)
            for batch in dev_dataloader:
                prompt_in = batch['prompt_in'].to(self.device)
                batch_length = batch['batch_length']
                encoded_raw_sent = batch['encoded_raw_sent'].to(self.device)

                outs, slots = self._generate(prompt_in, encoded_raw_sent, batch_length, length=self.args.gen_length)
                for out, slot in zip(outs, slots):
                    slot_label = label_map[domain_name][out.split('"')[-1].split('refers to')[0].strip()]
                    slot_texts = slot.split(';')
                    for text in slot_texts:
                        slot_values = text.replace(' .','').split()
                        pred_tags = self._reverse_labeling(original_seq_in, slot_values, slot_label, pred_tags)
            
            all_pred_tags.append(pred_tags)
            all_gold_tags.append(original_bio_seq_out)
        
        acc = precision_score(all_gold_tags, all_pred_tags)
        recall = recall_score(all_gold_tags, all_pred_tags)
        f1 = f1_score(all_gold_tags, all_pred_tags)

        if mode == 'test':
            logger.info('***********************************************************')
            logger.info(f'Target domain = {self.args.domain}')
            logger.info(f'n_samples = {self.args.nsamples}')
            logger.info(f'Test result: acc {acc:.4f}, recall {recall:.4f}, f1 {f1:.4f}')
            logger.info('***********************************************************')
        return acc, recall, f1



    def _reverse_labeling(self, tokens, value, slot_name, current_labels):
        assert len(tokens) == len(current_labels)

        v_tokens = [tk.lower().strip() for tk in value]
        tokens = [tk.lower().strip() for tk in tokens]


        def is_align(i):
            for j in range(len(v_tokens)):
                if not (i + j < len(tokens) and tokens[i + j] == v_tokens[j] and current_labels[i + j] == 'O'):
                    return False
            return True

        def fill_label(i):
            current_labels[i] = 'B-' + slot_name
            for j in range(1, len(v_tokens)):
                current_labels[i + j] = 'I-' + slot_name

        for ind, tk in enumerate(tokens):
            if is_align(ind):
                fill_label(ind)

        return current_labels



    def _generate(self, batch_prompt_in, encoded_raw_sent, each_in_length, length=1):
        original_each_in_length = [i for i in each_in_length]  # list[int]

        generated = batch_prompt_in  # [B, seq_len]
        with torch.no_grad():
            end_flags=torch.tensor([1] * batch_prompt_in.shape[0], device=self.device)
            for _ in range(length):
                inputs = {'input_ids': generated}
                outputs = self.model(**inputs)
                logits = outputs[1]  # [B, seq_len, vocab_size]

                t = []
                for i, j in enumerate(each_in_length):
                    t.append(logits[i].narrow(dim=0, start=j-1, length=1))  # [1, vocab_size]
                next_token_logits = torch.cat(t, dim=0)  # [B, vocab_size]
                next_token_logits = torch.gather(next_token_logits, 1, encoded_raw_sent)  # [B, raw_sent_len]
                next_token = torch.argmax(next_token_logits, dim=-1)  # [B]
                next_token.unsqueeze_(-1)  # [B, 1]
                next_token = torch.gather(encoded_raw_sent, 1, next_token)  # [B, 1]
                torch_next_token = next_token.squeeze(1)  # [B]
                next_token = torch_next_token.tolist()
                
                #collect '.',stop generating when all sentences in the batch meet '.'
                this_end = torch.where(torch_next_token==764, 0, 1)
                end_flags *= this_end

                g = []
                for i, j in enumerate(each_in_length):
                    p1 = generated[i].narrow(dim=0, start=0, length=j).unsqueeze(0)
                    p2 = torch.tensor([next_token[i]], device=self.device).unsqueeze(0)
                    p3 = generated[i].narrow(dim=0, start=j, length=generated.shape[-1] - j).unsqueeze(0)  # pad的字符
                    g.append(torch.cat([p1, p2, p3], dim=1))
                    each_in_length[i] += 1
                generated = torch.cat(g, dim=0)
                if torch.sum(end_flags).item() == 0:
                    break

        # split generated slots and input sentence
        generated = generated.cpu().tolist()
        texts = []
        slots = []
        for i, j in enumerate(generated):
            if 50256 in j:
                idx = j.index(50256)
                text = j[:idx]
            else:
                text = j
            texts.append(self.tokenizer.decode(text, clean_up_tokenization_spaces=False))
            slot = text[original_each_in_length[i]:]
            slots.append(self.tokenizer.decode(slot, clean_up_tokenization_spaces=False))
        return texts, slots



    def save_model(self, epoch):
        if not os.path.exists(os.path.join(self.args.model_dir, self.args.domain, str(self.args.nsamples))):
            os.makedirs(os.path.join(self.args.model_dir, self.args.domain, str(self.args.nsamples)))
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'args': self.args
        }
        torch.save(checkpoint, os.path.join(self.args.model_dir, self.args.domain, str(self.args.nsamples), 'model.pth'))
        logger.info("Saved model checkpoint to %s", os.path.join(self.args.model_dir, self.args.domain, str(self.args.nsamples), 'model.pth'))
    
    @classmethod
    def load_model(cls, model_dir, target_domain, n_samples):
        checkpoint = torch.load(os.path.join(model_dir, target_domain, str(n_samples), 'model.pth'))
        args = checkpoint['args']
        trainer = Trainer(args)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Model loaded from %s', os.path.join(model_dir, target_domain, str(n_samples), 'model.pth'))
        return trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train',action='store_true',help='Whether to train')
    parser.add_argument('--do_test',action='store_true',help='Whether to test')
    parser.add_argument("--model_dir", default='./model_dir', type=str, help="The directory to save the model")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16, help="Train batch_size")
    parser.add_argument('--dev_batch_size', type=int, default=16, help="Dev batch_size")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=6.25e-5)
    parser.add_argument("--warmup_rate", default=0.1, type=float, help="Linear warmup over warmup_steps.")
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--gen_length', type=int, default=40, help="max length of the generation utterances")
    parser.add_argument('--domain', default='AddToPlaylist', type=str)
    parser.add_argument('--nsamples', default=0, type=int)
    parser.add_argument('--log_dir', type=str, default='log_dir')
    parser.add_argument('--early_stop', default=10, type=int)


    args = parser.parse_args()
    args.log_file = os.path.join(args.log_dir, args.domain + '_' + str(args.nsamples) + '.log')

    init_logger(args.log_file)
    set_seed(args.seed)
    logger.info('**********************Job Start**********************')
    logger.info(f'Learning rate = {args.learning_rate}')
    logger.info(f'Warmup rate = {args.warmup_rate}')
    logger.info(f'Batch_size = {args.batch_size}')
    logger.info(f'Train epoch = {args.num_train_epochs}')
    logger.info(f'Domain = {args.domain}')
    logger.info(f'Nsamples = {args.nsamples}')

    trainer = Trainer(args)
    if args.do_train:
        trainer.train()
    if args.do_test:
        checkpoint = torch.load(os.path.join(args.model_dir, args.domain, str(args.nsamples), 'model.pth'))
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Starting test...')
        with torch.no_grad():
            trainer.evaluate('test')