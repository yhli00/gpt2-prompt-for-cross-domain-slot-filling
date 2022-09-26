import os


def read_row_data(domain, n_samples):
    '''
    return:
        train_set: List[dict]
            train_set[0]:
                {
                    'domain': str
                    'seq_in': list[str]
                    'seq_out': list[str]
                    'bio_seq_out': list[str]
                }
    '''
    dir_name = domain + '_' + str(n_samples)
    file_path = os.path.join('./data/snips', dir_name)
    train_path = os.path.join(file_path, 'train.tsv')
    dev_path = os.path.join(file_path, 'dev.tsv')
    test_path = os.path.join(file_path, 'test.tsv')
    with open(train_path, 'r') as f:
        lines = f.readlines()
    train_set = [{'domain': line.split('\t')[2].strip(), 'seq_in': line.split('\t')[0].split(), 'bio_seq_out': line.split('\t')[1].split(), 
                'seq_out': line.split('\t')[1].replace('I-', '').replace('B-', '').split()} for line in lines]
    with open(dev_path, 'r') as f:
        lines = f.readlines()
    dev_set = [{'domain': line.split('\t')[2].strip(), 'seq_in': line.split('\t')[0].split(), 'bio_seq_out': line.split('\t')[1].split(), 
                'seq_out': line.split('\t')[1].replace('I-', '').replace('B-', '').split()} for line in lines]
    with open(test_path, 'r') as f:
        lines = f.readlines()
    test_set = [{'domain': line.split('\t')[2].strip(), 'seq_in': line.split('\t')[0].split(), 'bio_seq_out': line.split('\t')[1].split(), 
                'seq_out': line.split('\t')[1].replace('I-', '').replace('B-', '').split()} for line in lines]
    return train_set, dev_set, test_set


if __name__ == '__main__':
    train_set, eval_set, test_set = read_row_data('AddToPlaylist', 20)
    print(len(train_set))
    print(len(eval_set))
    print(len(test_set))
    print(train_set[0])
    print(train_set[-1])