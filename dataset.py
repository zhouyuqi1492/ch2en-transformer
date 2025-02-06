import torch
from torch.utils.data import Dataset


class TranslationDataset(Dataset):

    def __init__(self, src_data_file_path, src_vocab_file_path,
                 tgt_data_file_path, tgt_vocab_file_path, max_len) -> None:
        super().__init__()
