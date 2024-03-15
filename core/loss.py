import torch
import torch.nn as nn


class SimCLRLoss(nn.Module):
    def __init__(self, temperature=0.5, n_aug: int = 2, device: str = "cuda"):
        super().__init__()
        self.temperature = temperature
        self.device = device
        self.n_aug = n_aug

    def forward(self, features1, features2):
        """
        See Also:
        https://github.com/sthalles/SimCLR/issues/29#issuecomment-1996853766

        Inspired by https://github.com/sthalles/SimCLR

        Args:
            features1 (torch.Tensor): augmented image representation 1 (vector)
            features2 (torch.Tensor): augmented image representation 2 (vector)

        Returns:
            torch.Tensor
        """
        batch_size = features1.shape[0]  # (B, D) -> B

        features1 = features1 / features1.norm(dim=1, keepdim=True)
        features2 = features2 / features2.norm(dim=1, keepdim=True)

        features = torch.concat([features1, features2], dim=0)  # (2N, D)
        similarity_matrix = torch.exp(
            torch.matmul(features, features.T) / self.temperature
        )

        labels = torch.cat([torch.arange(batch_size) for i in range(self.n_aug)], dim=0)

        # Remove diagonal (positive)
        labels = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels = labels.float()  # (2N, 2N) shape. Positive: (i,i), (i, i+N)
        mask = torch.eye(2 * batch_size, dtype=torch.bool)  # (2N, 2N)
        mask = mask.to(self.device)
        labels = labels[~mask].view(-1, 1)  # (2N*(2N-1), )
        similarity_matrix = similarity_matrix[~mask].view(-1, 1)

        positives = similarity_matrix[labels.bool()].view(-1, 1)  # (2N-1, 1)
        negatives = similarity_matrix[~labels.bool()].view(
            2 * batch_size, -1
        )  # (2N-1, 2N-1)

        logits = torch.concat(
            [positives, negatives], dim=1
        )  # (2N, 2N-1), First column is always positive

        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(
            self.device
        )  # (2N-1, )

        return torch.nn.functional.cross_entropy(logits, labels, reduction="mean")
