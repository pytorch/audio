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
        )

        costs = torch.FloatTensor(costs).to(device=device)
        gradients = torch.FloatTensor(gradients).to(device=device)
        ctx.grads = torch.autograd.Variable(gradients)

        return costs

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.view(-1, 1, 1, 1).to(ctx.grads)
        return ctx.grads.mul(grad_output), None, None, None, None, None, None, None, None

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
            alpha, alpha_cost = __class__.compute_alpha_one_sequence(
                log_probs=seq_log_probs, targets=seq_targets, blank=blank
            )

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
    ):
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return _NumpyTransducer.apply(
            log_probs,
            logit_lengths,
            target_lengths,
            targets,
            self.blank,
        )
