from collections import Counter

import torch
from torch import topk
from tqdm import tqdm


class GreedyIterableDecoder:
    def __init__(self, blank_label=0, collapse_repeated=True):
        self.blank_label = blank_label
        self.collapse_repeated = collapse_repeated

    def __call__(self, output):
        arg_maxes = torch.argmax(output, dim=-1)
        decodes = []
        for args in arg_maxes:
            decode = []
            for j, index in enumerate(args):
                if index != self.blank_label:
                    if self.collapse_repeated and j != 0 and index == args[j - 1]:
                        continue
                    decode.append(index.item())
            decode = torch.tensor(decode)
            decodes.append(decode)
        # decodes = torch.tensor(decodes)
        decodes = torch.nn.utils.rnn.pad_sequence(decodes, batch_first=True)
        return decodes


class GreedyDecoder:
    def __call__(self, outputs):
        """Greedy Decoder. Returns highest probability of class labels for each timestep

        Args:
            outputs (torch.Tensor): shape (input length, batch size, number of classes (including blank))

        Returns:
            torch.Tensor: class labels per time step.
        """
        _, indices = topk(outputs, k=1, dim=-1)
        return indices[..., 0]


def zeros_like(m):
    return zeros(len(m), len(m[0]))


def zeros(d1, d2):
    return list(list(0 for _ in range(d2)) for _ in range(d1))


def apply_transpose(f, m):
    return list(map(f, zip(*m)))


def argmax(l):
    return max(range(len(l)), key=lambda i: l[i])


def add1d2d(m1, m2):
    return [[v2 + v1 for v2 in m2_row] for m2_row, v1 in zip(m2, m1)]


def add1d1d(v1, v2):
    return [e + s for e, s in zip(v1, v2)]


class ListViterbiDecoder:
    def __init__(self, data_loader, vocab_size, n=2, progress_bar=False):
        self._transitions = self._build_transitions(
            data_loader, vocab_size, n, progress_bar
        )

    def __call__(self, emissions):
        return torch.tensor([self._decode(emissions[i].tolist(), self._transitions)[0] for i in range(len(emissions))])

    @staticmethod
    def _build_transitions(data_loader, vocab_size, n=2, progress_bar=False):

        # Count n-grams
        count = Counter()
        for _, label in tqdm(data_loader, disable=not progress_bar):
            count += Counter(a for a in zip(*(label[i:] for i in range(n))))

        # Write as matrix
        transitions = zeros(vocab_size, vocab_size)
        for (k1, k2), v in count.items():
            transitions[k1][k2] = v

        return transitions

    @staticmethod
    def _decode(emissions, transitions):
        scores = zeros_like(emissions)
        back_pointers = zeros_like(emissions)
        scores = emissions[0]

        # Generate most likely scores and paths for each step in sequence
        for i in range(1, len(emissions)):
            score_with_transition = add1d2d(scores, transitions)
            max_score_with_transition = apply_transpose(max, score_with_transition)
            scores = add1d1d(emissions[i], max_score_with_transition)
            back_pointers[i] = apply_transpose(argmax, score_with_transition)

        # Generate the most likely path
        viterbi = [argmax(scores)]
        for bp in reversed(back_pointers[1:]):
            viterbi.append(bp[viterbi[-1]])
        viterbi.reverse()
        viterbi_score = max(scores)

        return viterbi, viterbi_score


class ViterbiDecoder:
    def __init__(self, data_loader, vocab_size, n=2, progress_bar=False):
        self.vocab_size = vocab_size
        self.n = n
        self.top_k = 1
        self.progress_bar = progress_bar

        self._build_transitions(data_loader)

    def _build_transitions(self, data_loader):

        # Count n-grams

        c = Counter()
        for _, label in tqdm(data_loader, disable=not self.progress_bar):
            count = Counter(
                tuple(b.item() for b in a)
                for a in zip(*(label[i:] for i in range(self.n)))
            )
            c += count

        # Encode as transition matrix

        ind = torch.tensor([a for (a, _) in c.items()]).t()
        val = torch.tensor([b for (_, b) in c.items()], dtype=torch.float)

        transitions = (
            torch.sparse_coo_tensor(
                indices=ind, values=val, size=[self.vocab_size, self.vocab_size]
            )
            .coalesce()
            .to_dense()
        )
        transitions = transitions / torch.max(
            torch.tensor(1.0), transitions.max(dim=1)[0]
        ).unsqueeze(1)

        self.transitions = transitions

    def _viterbi_decode(self, tag_sequence: torch.Tensor):
        """
        Perform Viterbi decoding in log space over a sequence given a transition matrix
        specifying pairwise (transition) potentials between tags and a matrix of shape
        (sequence_length, num_tags) specifying unary potentials for possible tags per
        timestep.

        Parameters
        ----------
        tag_sequence : torch.Tensor, required.
            A tensor of shape (sequence_length, num_tags) representing scores for
            a set of tags over a given sequence.

        Returns
        -------
        viterbi_path : List[int]
            The tag indices of the maximum likelihood tag sequence.
        viterbi_score : float
            The score of the viterbi path.
        """
        sequence_length, num_tags = tag_sequence.size()

        path_scores = []
        path_indices = []
        # At the beginning, the maximum number of permutations is 1; therefore, we unsqueeze(0)
        # to allow for 1 permutation.
        path_scores.append(tag_sequence[0, :].unsqueeze(0))
        # assert path_scores[0].size() == (n_permutations, num_tags)

        # Evaluate the scores for all possible paths.
        for timestep in range(1, sequence_length):
            # Add pairwise potentials to current scores.
            # assert path_scores[timestep - 1].size() == (n_permutations, num_tags)
            summed_potentials = (
                path_scores[timestep - 1].unsqueeze(2) + self.transitions
            )
            summed_potentials = summed_potentials.view(-1, num_tags)

            # Best pairwise potential path score from the previous timestep.
            max_k = min(summed_potentials.size()[0], self.top_k)
            scores, paths = torch.topk(summed_potentials, k=max_k, dim=0)
            # assert scores.size() == (n_permutations, num_tags)
            # assert paths.size() == (n_permutations, num_tags)

            scores = tag_sequence[timestep, :] + scores
            # assert scores.size() == (n_permutations, num_tags)
            path_scores.append(scores)
            path_indices.append(paths.squeeze())

        # Construct the most likely sequence backwards.
        path_scores = path_scores[-1].view(-1)
        max_k = min(path_scores.size()[0], self.top_k)
        viterbi_scores, best_paths = torch.topk(path_scores, k=max_k, dim=0)

        viterbi_paths = []
        for i in range(max_k):

            viterbi_path = [best_paths[i].item()]
            for backward_timestep in reversed(path_indices):
                viterbi_path.append(int(backward_timestep.view(-1)[viterbi_path[-1]]))

            # Reverse the backward path.
            viterbi_path.reverse()

            # Viterbi paths uses (num_tags * n_permutations) nodes; therefore, we need to modulo.
            viterbi_path = [j % num_tags for j in viterbi_path]
            viterbi_paths.append(viterbi_path)

        return viterbi_paths, viterbi_scores

    def __call__(self, tag_sequence: torch.Tensor):

        outputs = []
        scores = []
        for i in range(tag_sequence.shape[1]):
            paths, score = self._viterbi_decode(tag_sequence[:, i, :])
            outputs.append(paths)
            scores.append(score)

        return torch.tensor(outputs).transpose(0, -1), torch.cat(scores)[:, 0, :]
