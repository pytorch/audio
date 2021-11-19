import math
from typing import Callable, Dict, List, Optional, NamedTuple, Tuple

import torch
from fairseq.data import Dictionary
from sentencepiece import SentencePieceProcessor

from .rnnt import RNNT


class Hypo(NamedTuple):
    tokens: List[int]
    predictor_out: torch.Tensor
    state: List[List[torch.Tensor]]
    score: float
    ali: List[int]
    blank: int
    key: str


def _batch_state(hypos: List[Hypo]) -> List[List[torch.Tensor]]:
    states: List[List[torch.Tensor]] = []
    for i in range(len(hypos[0].state)):
        batched_state_components: List[torch.Tensor] = []
        for j in range(len(hypos[0].state[i])):
            batched_state_components.append(
                torch.cat([hypo.state[i][j] for hypo in hypos])
            )
        states.append(batched_state_components)
    return states


def _slice_state(
    states: List[List[torch.Tensor]], idx: int, device: torch.device
) -> List[List[torch.Tensor]]:
    idx_tensor = torch.tensor([idx], device=device)
    return [
        [state.index_select(0, idx_tensor) for state in state_tuple]
        for state_tuple in states
    ]


def _default_hypo_sort_key(hypo: Hypo) -> float:
    return hypo.score / (len(hypo.tokens) + 1)


def _compute_updated_scores(
    hypos: List[Hypo],
    next_token_probs: torch.Tensor,
    beam_width: int,
    expand_beam: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    threshold_log_probs = (
        next_token_probs[:, :-1].max(dim=1).values.unsqueeze(1) - expand_beam
    )

    next_token_probs[:, :-1] = torch.where(
        next_token_probs[:, :-1] > threshold_log_probs,
        next_token_probs[:, :-1],
        torch.tensor(-99999, dtype=torch.float32),
    )

    hypo_scores = torch.tensor([h.score for h in hypos]).unsqueeze(1)
    nonblank_scores = (
        hypo_scores + next_token_probs[:, :-1]
    )  # [beam_width, num_tokens - 1]
    nonblank_nbest_scores, nonblank_nbest_idx = nonblank_scores.reshape(-1).topk(
        beam_width
    )

    nonblank_nbest_hypo_idx = nonblank_nbest_idx.div(
        nonblank_scores.shape[1], rounding_mode="trunc"
    )
    nonblank_nbest_token = nonblank_nbest_idx % nonblank_scores.shape[1]
    return nonblank_nbest_scores, nonblank_nbest_hypo_idx, nonblank_nbest_token


def _remove_hypo(hypo: Hypo, hypo_list: List[Hypo]) -> None:
    for i, elem in enumerate(hypo_list):
        if hypo.key == elem.key:
            del hypo_list[i]
            break


class RNNTBeamSearch(torch.nn.Module):
    def __init__(
        self,
        model: RNNT,
        blank: int,
        temperature: float = 1.0,
        expand_beam: int = 10,
        hypo_sort_key: Callable[[Hypo], float] = _default_hypo_sort_key,
        step_max_symbols: int = 100,
    ) -> None:
        super().__init__()
        self.model = model
        self.blank = blank
        self.temperature = temperature
        self.expand_beam = expand_beam
        self.hypo_sort_key = hypo_sort_key
        self.step_max_symbols = step_max_symbols

    def _init_b_hypos(
        self, device: torch.device, prev_hypo: Optional[Hypo]
    ) -> List[Hypo]:
        if prev_hypo is not None:
            prev_token = torch.tensor([[prev_hypo.tokens[-1]]], device=device)
        else:
            prev_token = torch.tensor([[self.blank]], device=device)

        one_tensor = torch.tensor([1], device=device)
        pred_out, _, pred_states = self.model.predict(
            prev_token, one_tensor, prev_hypo.state if prev_hypo is not None else None,
        )
        blank_hypo = Hypo(
            tokens=[self.blank],
            predictor_out=pred_out[0].detach(),
            state=pred_states,
            score=0.0,
            ali=[-1],
            blank=self.blank,
            key=str([self.blank]),
        )
        return [blank_hypo]

    def _gen_next_token_probs(
        self, enc_out: torch.Tensor, hypos: List[Hypo], device: torch.device
    ) -> torch.Tensor:
        one_tensor = torch.tensor([1], device=device)
        predictor_out = torch.stack([h.predictor_out for h in hypos], dim=0)
        joined_out, src_lengths, tgt_lengths = self.model.join(
            enc_out,
            one_tensor,
            predictor_out,
            torch.tensor([1] * len(hypos), device=device),
        )  # [beam_width, 1, 1, num_tokens]
        joined_out = torch.nn.functional.log_softmax(
            joined_out / self.temperature, dim=3
        )
        joined_out[:, :, :, :4].add_(-99999)  # Blank out invalid tokens
        return joined_out[:, 0, 0]

    def _gen_b_hypos(
        self,
        b_hypos: List[Hypo],
        a_hypos: List[Hypo],
        next_token_probs: torch.Tensor,
        key_to_b_hypo: Dict[str, Hypo],
    ) -> List[Hypo]:
        for i in range(len(a_hypos)):
            h_a = a_hypos[i]
            append_blank_score = h_a.score + next_token_probs[i, -1]
            if h_a.key in key_to_b_hypo:
                h_b = key_to_b_hypo[h_a.key]
                ali = h_a.ali if h_b.score < h_a.score else h_b.ali
                _remove_hypo(h_b, b_hypos)
                h_b = Hypo(
                    tokens=h_a.tokens,
                    predictor_out=h_a.predictor_out,
                    state=h_a.state,
                    score=float(torch.tensor(h_b.score).logaddexp(append_blank_score)),
                    ali=ali,
                    blank=self.blank,
                    key=str(h_a.tokens),
                )
                b_hypos.append(h_b)
                key_to_b_hypo[h_a.key] = h_b
            else:
                h_b = Hypo(
                    tokens=h_a.tokens,
                    predictor_out=h_a.predictor_out,
                    state=h_a.state,
                    score=float(append_blank_score),
                    ali=h_a.ali,
                    blank=self.blank,
                    key=str(h_a.tokens),
                )
                b_hypos.append(h_b)
                key_to_b_hypo[h_b.key] = h_b
        _, sorted_idx = torch.tensor([hypo.score for hypo in b_hypos]).sort()
        return [b_hypos[idx] for idx in sorted_idx]

    def _gen_a_hypos(
        self,
        a_hypos: List[Hypo],
        b_hypos: List[Hypo],
        next_token_probs: torch.Tensor,
        t: int,
        beam_width: int,
        device: torch.device,
    ) -> List[Hypo]:
        (
            nonblank_nbest_scores,
            nonblank_nbest_hypo_idx,
            nonblank_nbest_token,
        ) = _compute_updated_scores(
            a_hypos, next_token_probs, beam_width, self.expand_beam, device,
        )

        if len(b_hypos) < beam_width:
            b_nbest_score = -float("inf")
        else:
            b_nbest_score = b_hypos[-beam_width].score

        base_hypos: List[Hypo] = []
        new_tokens: List[int] = []
        new_scores: List[float] = []
        for i in range(beam_width):
            score = float(nonblank_nbest_scores[i])
            if score > b_nbest_score:
                a_hypo_idx = int(nonblank_nbest_hypo_idx[i])
                base_hypos.append(a_hypos[a_hypo_idx])
                new_tokens.append(int(nonblank_nbest_token[i]))
                new_scores.append(score)

        if base_hypos:
            new_hypos = self._gen_new_hypos(
                base_hypos, new_tokens, new_scores, t, device
            )
        else:
            new_hypos: List[Hypo] = []

        return new_hypos

    def _gen_new_hypos(
        self,
        base_hypos: List[Hypo],
        tokens: List[int],
        scores: List[float],
        t: int,
        device: torch.device,
    ) -> List[Hypo]:
        tgt_tokens = torch.tensor([[token] for token in tokens], device=device)
        states = _batch_state(base_hypos)
        pred_out, _, pred_states = self.model.predict(
            tgt_tokens, torch.tensor([1] * len(base_hypos), device=device), states,
        )
        new_hypos: List[Hypo] = []
        for i, h_a in enumerate(base_hypos):
            new_tokens = h_a.tokens + [tokens[i]]
            new_hypos.append(
                Hypo(
                    tokens=new_tokens,
                    predictor_out=pred_out[i].detach(),
                    state=_slice_state(pred_states, i, device),
                    score=scores[i],
                    ali=h_a.ali + [t],
                    blank=self.blank,
                    key=str(new_tokens),
                )
            )
        return new_hypos

    def _search(
        self,
        input: torch.Tensor,
        length: torch.Tensor,
        state: Optional[List[List[torch.Tensor]]],
        prev_hypo: Optional[Hypo],
        beam_width: int,
        streaming_mode: bool,
    ) -> Tuple[List[Hypo], Optional[List[List[torch.Tensor]]]]:
        if streaming_mode:
            enc_out, _, state = self.model.transcribe_streaming(input, length, state)
        else:
            enc_out, _ = self.model.transcribe(input, length)
            state = None

        n_time_steps = enc_out.shape[1]
        device = input.device

        a_hypos: List[Hypo] = []
        b_hypos = self._init_b_hypos(device, prev_hypo)
        for t in range(n_time_steps):
            a_hypos = b_hypos
            b_hypos = torch.jit.annotate(List[Hypo], [])
            key_to_b_hypo: Dict[str, Hypo] = {}
            symbols_current_t = 0

            while a_hypos:
                next_token_probs = self._gen_next_token_probs(
                    enc_out[:, t : t + 1], a_hypos, device
                )
                next_token_probs = next_token_probs.cpu()
                b_hypos = self._gen_b_hypos(
                    b_hypos, a_hypos, next_token_probs, key_to_b_hypo,
                )

                if symbols_current_t == self.step_max_symbols:
                    break

                a_hypos = self._gen_a_hypos(
                    a_hypos, b_hypos, next_token_probs, t, beam_width, device,
                )
                if a_hypos:
                    symbols_current_t += 1

            _, sorted_idx = torch.tensor(
                [self.hypo_sort_key(hypo) for hypo in b_hypos]
            ).topk(beam_width)
            b_hypos = [b_hypos[idx] for idx in sorted_idx]

        return b_hypos, state

    def forward(
        self, input: torch.Tensor, length: torch.Tensor, beam_width: int
    ) -> List[Hypo]:
        r"""Performs beam search for the given input sequence.

        T: number of frames;
        D: feature dimension of each frame.

        Args:
            input (torch.Tensor): sequence of input frames, with shape (1, T, D).
            length (torch.Tensor): number of valid frames in input
                sequence, with shape (1,).
            beam_width (int): beam size to use during search.

        Returns:
            List[Hypo]: top-`beam_width` hypotheses found by beam search.
        """
        assert (
            len(input.shape) == 3 and input.shape[0] == 1
        ), "input must be of shape (1, T, D)"
        assert length.shape == (1,), "length must be of shape (1,)"
        hypos, _ = self._search(input, length, None, None, beam_width, False)
        return hypos

    @torch.jit.export
    def infer(
        self,
        input: torch.Tensor,
        length: torch.Tensor,
        state: Optional[List[List[torch.Tensor]]],
        prev_hypo: Optional[Hypo],
        beam_width: int,
    ) -> Tuple[List[Hypo], Optional[List[List[torch.Tensor]]]]:
        r"""Performs beam search for the given input sequence in streaming mode.

        T: number of frames;
        D: feature dimension of each frame.

        Args:
            input (torch.Tensor): sequence of input frames, with shape (1, T, D).
            length (torch.Tensor): number of valid frames in input
                sequence, with shape (1,).
            state (List[List[torch.Tensor]] or None): list of lists of tensors
                representing transcription network internal state generated in preceding
                invocation of ``infer``.
            prev_hypo (Hypo or None): hypothesis from preceding invocation to seed search with.
            beam_width (int): beam size to use during search.

        Returns:
            List[Hypo]: top-`beam_width` hypotheses found by beam search.
        """
        assert (
            len(input.shape) == 3 and input.shape[0] == 1
        ), "input must be of shape (1, T, D)"
        assert length.shape == (1,), "length must be of shape (1,)"
        return self._search(input, length, state, prev_hypo, beam_width, True)


def post_process_hypos(
    hypos: List[Hypo], sp_model: SentencePieceProcessor, tgt_dict: Dictionary
) -> List[Tuple[str, float, List[int], List[int]]]:
    post_process_remove_list = [
        sp_model.unk_id(),
        sp_model.eos_id(),
        sp_model.pad_id(),
    ]
    hypos_str = [
        tgt_dict.string(
            [
                token_index
                for token_index in h.tokens[1:]
                if token_index not in post_process_remove_list
            ]
        )
        for h in hypos
    ]
    hypos_str = [sp_model.DecodePieces(s.split()) for s in hypos_str]
    hypos_ali = [h.ali[1:] for h in hypos]
    hypos_ids = [h.tokens[1:] for h in hypos]
    hypos_score = [[math.exp(h.score)] for h in hypos]

    nbest_batch = list(zip(hypos_str, hypos_score, hypos_ali, hypos_ids))

    return nbest_batch
