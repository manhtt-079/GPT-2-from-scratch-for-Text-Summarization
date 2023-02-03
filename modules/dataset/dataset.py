import pandas as pd
import torch
from torch.utils.data import DataLoader, IterableDataset, Dataset

class JPDataset(Dataset):
    def __init__(self,
                 tokenizer,
                 mode='train',
                 file_path: str = './data/japanese_text_sum.csv',
                 max_seq_length: int = 512,
                 ) -> None:

        super(JPDataset, self).__init__()

        self.mode = mode
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.data = pd.read_csv(file_path)
        if mode == 'train':
            self.data = self.data[self.data.is_train]
        elif mode == 'valid':
            self.data = self.data[self.data.is_val]
        else:
            self.data = self.data[self.data.is_test]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        input_ids = self.tokenizer.encode(
            row['source']+row['target'], max_length=self.max_seq_length, truncation=True, padding='max_length')

        return {
            'input_ids': torch.tensor(input_ids),
            'label': torch.tensor(input_ids)
        }


class JPIterDataset(IterableDataset):
    def __init__(self,
                 tokenizer,
                 mode='train',
                 file_path: str = './data/japanese_text_sum.csv',
                 max_seq_length: int = 512,
                 random_state: int = 42,
                 ) -> None:

        super(JPIterDataset, self).__init__()

        self.mode = mode
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.data = pd.read_csv(file_path)
        if mode == 'train':
            self.data = self.data[self.data.is_train]
            self.data = self.data.sample(frac=1, random_state=random_state)
        elif mode == 'valid':
            self.data = self.data[self.data.is_val]
        else:
            self.data = self.data[self.data.is_test]

    def __len__(self) -> int:
        return len(self.data)
    
    def __iter__(self):
        for src, tgt in zip(self.data.source, self.data.target):
            input_ids = self.tokenizer.encode(src+tgt, max_length=self.max_seq_length, truncation=True, padding='max_length')
            
            yield {
                'input_ids': torch.tensor(input_ids),
                'label': torch.tensor(input_ids)
            }


def dataset(tokenizer,
            mode='train',
            file_path: str = "./data/",
            max_seq_length: int=512,
            shuffle: bool = False,
            batch_size: int = 4,
            num_workers: int = 16) -> DataLoader:

    if mode not in ['train', 'valid', 'test']:
        raise ValueError("`mode` must be in: ['train', 'valid', 'test']")

    dataset = JPDataset(
        tokenizer=tokenizer,
        mode=mode,
        file_path=file_path,
        max_seq_length=max_seq_length
    )
    return DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)