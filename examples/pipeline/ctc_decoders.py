from tqdm import tqdm
from collections import Counter

import torch
from torch import topk


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
            count = zip([label[i:].item() for i in range(self.n)])
            count = Counter(*count)
            c += count

        # Encode as transition matrix

        ind = torch.tensor(list(zip(*(a for (a, _) in c.items()))))
        val = torch.tensor((b for (_, b) in c.items()), dtype=torch.float)

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
                path_scores[timestep - 1].unsqueeze(2) + self.transition_matrix
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
            paths, score = self._viterbi_decode(
                tag_sequence[:, i, :], self._transitions
            )
            outputs.append(paths)
            scores.append(score)

        return torch.tensor(outputs).transpose(0, -1), torch.cat(scores)[:, 0, :]
