import numpy as np
import torch


class _NumpyTransducer(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        log_probs,
        logit_lengths,
        target_lengths,
        targets,
        blank=-1,
        wordpiece_ends=None,
        left_buffer=0,
        right_buffer=0,
    ):
        device = log_probs.device
        log_probs = log_probs.cpu().data.numpy()
        logit_lengths = logit_lengths.cpu().data.numpy()
        target_lengths = target_lengths.cpu().data.numpy()
        targets = targets.cpu().data.numpy()

        gradients, costs, _, _ = __class__.compute(
            log_probs=log_probs,
            logit_lengths=logit_lengths,
            target_lengths=target_lengths,
            targets=targets,
            blank=blank,
            wordpiece_ends=wordpiece_ends,
            left_buffer=left_buffer,
            right_buffer=right_buffer,
        )

        costs = torch.FloatTensor(costs).to(device=device)
        gradients = torch.FloatTensor(gradients).to(device=device)
        ctx.grads = torch.autograd.Variable(gradients)

        return costs

    @staticmethod
    def backward(ctx, output_gradients):
        return ctx.grads, None, None, None, None, None, None, None, None

    @staticmethod
    def compute_alpha_one_sequence(log_probs, targets, blank=-1):
        max_T, max_U, D = log_probs.shape
        alpha = np.zeros((max_T, max_U), dtype=np.float32)
        for t in range(1, max_T):
            alpha[t, 0] = alpha[t - 1, 0] + log_probs[t - 1, 0, blank]

        for u in range(1, max_U):
            alpha[0, u] = alpha[0, u - 1] + log_probs[0, u - 1, targets[u - 1]]

        for t in range(1, max_T):
            for u in range(1, max_U):
                skip = alpha[t - 1, u] + log_probs[t - 1, u, blank]
                emit = alpha[t, u - 1] + log_probs[t, u - 1, targets[u - 1]]
                alpha[t, u] = np.logaddexp(skip, emit)

        cost = -(alpha[-1, -1] + log_probs[-1, -1, blank])
        return alpha, cost

    @staticmethod
    def compute_alpha_one_sequence_restricted(
        log_probs, targets, wordpiece_ends, blank=-1, left_buffer=0, right_buffer=0
    ):
        # Note - this is a more generic case
        # If we set left_buffer = right_buffer = inf this becomes same as
        # `compute_alpha_one_sequence`. However keeping both versions,
        # since without the checks, compute_alpha_one_sequence may be tiny
        # bit faster than `compute_alpha_one_sequence_restricted`

        max_T, max_U, D = log_probs.shape
        check = AlignmentRestrictionCheck(
            max_U, max_T, wordpiece_ends, left_buffer, right_buffer
        )

        alpha = np.zeros((max_T, max_U), dtype=np.float32)
        alpha.fill(float("-inf"))
        alpha[0, 0] = 0.0
        for t in range(1, max_T):
            if not check.alpha_blank_transition(t, 0):
                break
            alpha[t, 0] = log_probs[t - 1, 0, blank] + alpha[t - 1, 0]

        for u in range(1, max_U):
            if not check.alpha_emit_transition(0, u):
                break
            alpha[0, u] = log_probs[0, u - 1, targets[u - 1]] + alpha[0, u - 1]

        for u in range(1, max_U):
            start_t, end_t = check.valid_time_ranges(u)
            for t in range(start_t, end_t + 1):
                skip, emit = float("-inf"), float("-inf")

                if check.alpha_blank_transition(t, u):
                    skip = alpha[t - 1, u] + log_probs[t - 1, u, blank]

                if check.alpha_emit_transition(t, u):
                    emit = alpha[t, u - 1] + log_probs[t, u - 1, targets[u - 1]]

                if emit != float("-inf") or skip != float("-inf"):
                    alpha[t, u] = np.logaddexp(skip, emit)

        cost = -(alpha[-1, -1] + log_probs[-1, -1, blank])
        return alpha, cost

    @staticmethod
    def compute_beta_one_sequence(log_probs, targets, blank=-1):
        max_T, max_U, D = log_probs.shape
        beta = np.zeros((max_T, max_U), dtype=np.float32)
        beta[-1, -1] = log_probs[-1, -1, blank]

        for t in reversed(range(max_T - 1)):
            beta[t, -1] = beta[t + 1, -1] + log_probs[t, -1, blank]

        for u in reversed(range(max_U - 1)):
            beta[-1, u] = beta[-1, u + 1] + log_probs[-1, u, targets[u]]

        for t in reversed(range(max_T - 1)):
            for u in reversed(range(max_U - 1)):
                skip = beta[t + 1, u] + log_probs[t, u, blank]
                emit = beta[t, u + 1] + log_probs[t, u, targets[u]]
                beta[t, u] = np.logaddexp(skip, emit)

        cost = -beta[0, 0]
        return beta, cost

    @staticmethod
    def compute_beta_one_sequence_restricted(
        log_probs, targets, wordpiece_ends, blank=-1, left_buffer=0, right_buffer=0
    ):
        # Note - this is a more generic case
        # If we set left_buffer = right_buffer = inf this becomes same as
        # `compute_beta_one_sequence`. However keeping both versions,
        # since without the checks, compute_beta_one_sequence may be tiny
        # bit faster than `compute_beta_one_sequence_restricted`
        max_T, max_U, D = log_probs.shape

        check = AlignmentRestrictionCheck(
            max_U, max_T, wordpiece_ends, left_buffer, right_buffer
        )
        beta = np.zeros((max_T, max_U), dtype=np.float32)
        beta.fill(float("-inf"))
        beta[-1, -1] = log_probs[-1, -1, blank]

        for t in reversed(range(max_T - 1)):
            if not check.beta_blank_transition(t, max_U - 1):
                break
            beta[t, -1] = beta[t + 1, -1] + log_probs[t, -1, blank]

        for u in reversed(range(max_U - 1)):
            if not check.beta_emit_transition(max_T - 1, u):
                break
            beta[-1, u] = beta[-1, u + 1] + log_probs[-1, u, targets[u]]

        for u in reversed(range(max_U - 1)):
            start_t, end_t = check.valid_time_ranges(u)
            for t in reversed(range(start_t, end_t + 1)):
                skip, emit = float("-inf"), float("-inf")

                if check.beta_emit_transition(t, u):
                    emit = beta[t, u + 1] + log_probs[t, u, targets[u]]

                if check.beta_blank_transition(t, u):
                    skip = beta[t + 1, u] + log_probs[t, u, blank]

                if emit != float("-inf") or skip != float("-inf"):
                    beta[t, u] = np.logaddexp(skip, emit)

        cost = -beta[0, 0]
        return beta, cost

    @staticmethod
    def compute_gradients_one_sequence(
        log_probs, alpha, beta, targets, blank=-1
    ):
        max_T, max_U, D = log_probs.shape
        gradients = np.full(log_probs.shape, float("-inf"))
        cost = -beta[0, 0]

        gradients[-1, -1, blank] = alpha[-1, -1]

        gradients[:-1, :, blank] = alpha[:-1, :] + beta[1:, :]

        for u, l in enumerate(targets):
            gradients[:, u, l] = alpha[:, u] + beta[:, u + 1]

        gradients = -(np.exp(gradients + log_probs + cost))
        return gradients

    @staticmethod
    def compute(
        log_probs,
        logit_lengths,
        target_lengths,
        targets,
        blank=-1,
        wordpiece_ends=None,
        left_buffer=0,
        right_buffer=0,
    ):
        gradients = np.zeros_like(log_probs)
        B_tgt, max_T, max_U, D = log_probs.shape
        B_src = logit_lengths.shape[0]

        H = int(B_tgt / B_src)

        alphas = np.zeros((B_tgt, max_T, max_U))
        betas = np.zeros((B_tgt, max_T, max_U))
        betas.fill(float("-inf"))
        alphas.fill(float("-inf"))
        costs = np.zeros(B_tgt)
        for b_tgt in range(B_tgt):
            b_src = int(b_tgt / H)
            T = int(logit_lengths[b_src])
            # NOTE: see https://arxiv.org/pdf/1211.3711.pdf Section 2.1
            U = int(target_lengths[b_tgt]) + 1

            seq_log_probs = log_probs[b_tgt, :T, :U, :]
            seq_targets = targets[b_tgt, : int(target_lengths[b_tgt])]
            if wordpiece_ends is not None:
                seq_wordpiece_ends = wordpiece_ends[b_tgt, : int(target_lengths[b_tgt]) + 1]
                alpha, alpha_cost = __class__.compute_alpha_one_sequence_restricted(
                    log_probs=seq_log_probs,
                    targets=seq_targets,
                    blank=blank,
                    wordpiece_ends=seq_wordpiece_ends,
                    left_buffer=left_buffer,
                    right_buffer=right_buffer,
                )
            else:
                alpha, alpha_cost = __class__.compute_alpha_one_sequence(
                    log_probs=seq_log_probs, targets=seq_targets, blank=blank
                )

            if wordpiece_ends is not None:
                seq_wordpiece_ends = wordpiece_ends[b_tgt, : int(target_lengths[b_tgt]) + 1]
                beta, beta_cost = __class__.compute_beta_one_sequence_restricted(
                    log_probs=seq_log_probs,
                    targets=seq_targets,
                    blank=blank,
                    wordpiece_ends=seq_wordpiece_ends,
                    left_buffer=left_buffer,
                    right_buffer=right_buffer,
                )
            else:
                beta, beta_cost = __class__.compute_beta_one_sequence(
                    log_probs=seq_log_probs, targets=seq_targets, blank=blank
                )

            seq_gradients = __class__.compute_gradients_one_sequence(
                log_probs=seq_log_probs,
                alpha=alpha,
                beta=beta,
                targets=seq_targets,
                blank=blank,
            )
            np.testing.assert_almost_equal(alpha_cost, beta_cost, decimal=2)
            gradients[b_tgt, :T, :U, :] = seq_gradients
            costs[b_tgt] = beta_cost
            alphas[b_tgt, :T, :U] = alpha
            betas[b_tgt, :T, :U] = beta

        return gradients, costs, alphas, betas


class NumpyTransducerLoss(torch.nn.Module):
    def __init__(self, blank=-1):
        super().__init__()
        self.blank = blank

    def forward(
        self,
        logits,
        logit_lengths,
        target_lengths,
        targets,
        wordpiece_ends=None,
        left_buffer=0,
        right_buffer=0,
    ):
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return _NumpyTransducer.apply(
            log_probs,
            logit_lengths,
            target_lengths,
            targets,
            self.blank,
            wordpiece_ends,
            left_buffer,
            right_buffer,
        )


class AlignmentRestrictionCheck(object):
    """
    Helper class which returns locations where
    blank / emit transitions are valid while
    updating alpha / betas for Alignment
    Restricted RNN Transducer loss. This class
    will help test c++ code, the correctness has been
    verified by thorough manual inspections in n268104

    """

    def __init__(self, U, T, wordpiece_ends, left_buffer, right_buffer):
        self.U = U
        self.T = T
        # assume word_ends contains the ground truth end
        # timestep of every word piece.
        # assume 0th wordpiece_ends is <bos> and its endtime is 0
        assert wordpiece_ends[0] == 0
        assert len(wordpiece_ends) == U
        self.wordpiece_ends = wordpiece_ends
        self.left_buffer = left_buffer
        self.right_buffer = right_buffer

    def valid_time_ranges(self, u):
        # Returns ranges of valid paths which satisfy
        # word boundary constraints
        # For bos ranges are valid from 0 until end time of
        # 1st symbol + right_buffer
        start = max(self.wordpiece_ends[u] - self.left_buffer, 0)
        if u == self.U - 1:
            end = self.T - 1
        else:
            end = min(self.wordpiece_ends[u + 1] + self.right_buffer, self.T - 1)

        return start, end

    def alpha_blank_transition(self, t, u):
        # Examine if doing blank transition into (t, u)
        # is allowed while updating alphas.
        if u == 0 and t == 0:
            return False

        # For alphas blank transitions are valid from:
        # start time when current symbol is emitted
        start = max(self.wordpiece_ends[u] - self.left_buffer + 1, 1)

        # For alphas blank transitions are valid until
        # last timestep we can emit next symbol
        if u == self.U - 1:
            # for last symbol, allow blank transitions till end
            end = self.T - 1
        else:
            end = min(self.wordpiece_ends[u + 1] + self.right_buffer, self.T - 1)

        return start <= t and t <= end

    def alpha_emit_transition(self, t, u):
        # Examine if doing emit transition into (t, u)
        # is allowed while updating alphas.
        if u == 0:
            # emit transitions into (t, 0) for alpha are invalid
            return False

        # For alphas emit are valid starting from:
        # first time when current symbol can be emitted
        start = max(self.wordpiece_ends[u] - self.left_buffer, 0)

        # For alphas emit are valid until:
        # last timestep when current symbol can be emitted
        end = min(self.wordpiece_ends[u] + self.right_buffer, self.T - 1)

        return start <= t and t <= end

    def beta_blank_transition(self, t, u):
        # Examine if doing blank transition into (t, u)
        # is allowed while updating betas.

        # for beta, blanks transitions are can start
        # first timestep when we emit previous symbol
        start = max(self.wordpiece_ends[u] - self.left_buffer, 0)

        # for beta, blanks transitions are valid until
        # last timestep we can emit current symbol
        # note: T-2, we init beta[-1, -1] by log_prob[-1, -1, blank]
        if u == self.U - 1:
            end = self.T - 2
        else:
            # we cannot transition into beta for T-1
            end = min(self.wordpiece_ends[u + 1] + self.right_buffer - 1, self.T - 2)

        return start <= t and t <= end

    def beta_emit_transition(self, t, u):
        # Examine if doing emit transition into (t, u)
        # is allowed while updating betas

        # for last symbol, we do not allow emit transition,
        # so beta into u-1, t is invalid
        if u == self.U - 1:
            return False

        # For betas we allow emit transition starting from
        # first time we can emit next symbol
        start = max(0, self.wordpiece_ends[u + 1] - self.left_buffer)

        # For betas we allow emit transitions to end with
        # last time we can emit next symbol
        end = min(self.wordpiece_ends[u + 1] + self.right_buffer, self.T - 1)

        return start <= t and t <= end
