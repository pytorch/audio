import torch


class Decoder(torch.nn.Module):
    def __init__(self, labels):
        super().__init__()
        self.labels = labels

    def forward(self, logits: torch.Tensor) -> str:
        """Given a sequence logits over labels, get the best path string

        Args:
            logits (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
            str: The resulting transcript
        """
        best_path = torch.argmax(logits, dim=-1)  # [num_seq,]
        best_path = torch.unique_consecutive(best_path, dim=-1)
        hypothesis = ""
        for i in best_path:
            char = self.labels[i]
            if char in ["<s>", "<pad>"]:
                continue
            if char == "|":
                char = " "
            hypothesis += char
        return hypothesis
