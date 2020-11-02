import collections
import itertools


class LanguageModel:
    def __init__(self, labels, char_blank, char_space):

        self.char_space = char_space
        self.char_blank = char_blank

        labels = [l for l in labels]
        self.length = len(labels)
        enumerated = list(enumerate(labels))
        flipped = [(sub[1], sub[0]) for sub in enumerated]

        d1 = collections.OrderedDict(enumerated)
        d2 = collections.OrderedDict(flipped)
        self.mapping = {**d1, **d2}

    def encode(self, iterable):
        if isinstance(iterable, list):
            return [self.encode(i) for i in iterable]
        else:
            return [self.mapping[i] + self.mapping[self.char_blank] for i in iterable]

    def decode(self, tensor):
        if len(tensor) > 0 and isinstance(tensor[0], list):
            return [self.decode(t) for t in tensor]
        else:
            # not idempotent, since clean string
            x = (self.mapping[i] for i in tensor)
            x = "".join(i for i, _ in itertools.groupby(x))
            x = x.replace(self.char_blank, "")
            # x = x.strip()
            return x

    def __len__(self):
        return self.length
