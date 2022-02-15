class MockSentencePieceProcessor:
    def __init__(self, num_symbols, *args, **kwargs):
        self.num_symbols = num_symbols

    def get_piece_size(self):
        return self.num_symbols

    def encode(self, input):
        return [1, 5, 2]

    def decode(self, input):
        return "hey"

    def unk_id(self):
        return 0

    def eos_id(self):
        return 1

    def pad_id(self):
        return 2


class MockCustomDataset:
    def __init__(self, base_dataset, *args, **kwargs):
        self.base_dataset = base_dataset

    def __getitem__(self, n: int):
        return [self.base_dataset[n]]

    def __len__(self):
        return len(self.base_dataset)
