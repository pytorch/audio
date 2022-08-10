import torch
from torchaudio.models.decoder import CTCDecoderLM, CTCDecoderLMState
from torchaudio.models.decoder._ctc_decoder import _create_word_dict, _Dictionary, _KenLM, _load_words


class CustomZeroLM(CTCDecoderLM):
    def __init__(self):
        CTCDecoderLM.__init__(self)

    def start(self, start_with_nothing: bool):
        return CTCDecoderLMState()

    def score(self, state: CTCDecoderLMState, token_index: int):
        return (state.child(token_index), 0.0)

    def finish(self, state: CTCDecoderLMState):
        return (state, 0.0)


class CustomKenLM(CTCDecoderLM):
    def __init__(self, kenlm_file, dict_file):
        CTCDecoderLM.__init__(self)
        kenlm_dict = _create_word_dict(_load_words(dict_file))
        self.model = _KenLM(kenlm_file, kenlm_dict)

    def start(self, start_with_nothing: bool):
        return self.model.start(start_with_nothing)

    def score(self, state: CTCDecoderLMState, token_index: int):
        return self.model.score(state, token_index)

    def finish(self, state: CTCDecoderLMState):
        return self.model.finish(state)


class BiasedLM(torch.nn.Module):
    def __init__(self, dict_file, keyword):
        super(BiasedLM, self).__init__()
        self.dictionary = _Dictionary(dict_file)
        self.keyword = keyword

    def forward(self, token_idx):
        if self.dictionary.get_entry(token_idx) == self.keyword:
            return torch.tensor(10)
        elif self.dictionary.get_entry(token_idx) == "<unk>":
            return torch.tensor(-torch.inf)
        return torch.tensor(-10)


class CustomBiasedLM(CTCDecoderLM):
    def __init__(self, model, dict_file):
        CTCDecoderLM.__init__(self)
        self.model = model
        self.vocab = _Dictionary(dict_file)
        self.eos = self.vocab.get_index("|")
        self.states = {}

        model.eval()

    def start(self, start_with_nothing: bool = False):
        state = CTCDecoderLMState()
        with torch.no_grad():
            score = self.model(self.eos)

        self.states[state] = score
        return state

    def score(self, state: CTCDecoderLMState, token_index: int):
        outstate = state.child(token_index)
        if outstate not in self.states:
            score = self.model(token_index)
            self.states[outstate] = score
        score = self.states[outstate]

        return outstate, score

    def finish(self, state: CTCDecoderLMState):
        return self.score(state, self.eos)
