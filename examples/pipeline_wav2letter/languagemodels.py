import itertools
from collections.abc import Iterable


class LanguageModel:
    def __init__(self, labels, char_blank, char_space):

        self.char_space = char_space
        self.char_blank = char_blank

        enumerated = list(enumerate(labels))
        flipped = [(sub[1], sub[0]) for sub in enumerated]

        self._decode_map = dict(enumerated)
        self._encode_map = dict(flipped)

    def encode(self, listlike):
        if not isinstance(listlike, str):
            return [self.encode(i) for i in listlike]
        return [self._encode_map[i] + self._encode_map[self.char_blank] for i in listlike]

    def decode(self, tensor):
        if len(tensor) > 0 and isinstance(tensor[0], Iterable):
            return [self.decode(t) for t in tensor]

        # not idempotent, since clean string
        x = (self._decode_map[i] for i in tensor)
        x = "".join(i for i, _ in itertools.groupby(x))
        x = x.replace(self.char_blank, "")
        # x = x.strip()
        return x

    def __len__(self):
        return len(self._encode_map)
