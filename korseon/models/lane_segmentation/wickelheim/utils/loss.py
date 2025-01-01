import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        loss = - (1 - pt) ** self.gamma * torch.log(pt)

        if self.alpha is not None:
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class DiscriminativeLoss(nn.Module):
    def __init__(self, delta_var=0.5, delta_dist=1.5, norm=2, alpha=1.0, beta=1.0, gamma=0.001, reduction='mean'):
        super(DiscriminativeLoss, self).__init__()
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.norm = norm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, embeddings, instance_masks):
        num_samples = embeddings.size(0)
        loss = 0.0

        for i in range(num_samples):
            loss += self._discriminative_loss_single(embeddings[i], instance_masks[i])

        return loss / num_samples

    def _discriminative_loss_single(self, embedding, instance_mask):
        num_clusters = instance_mask.size(0)  # Number of clusters is the first dimension
        if num_clusters == 0:
            return 0.0  # No clusters, no loss

        embedding_dim = embedding.size(0)
        mask_mean = []

        # Calculate mean embeddings for each cluster
        for cluster_id in range(num_clusters):
            mask = instance_mask[cluster_id].unsqueeze(0).float()  # Mask for the current cluster

            # Check if mask contains any pixels
            if torch.sum(mask) == 0:
                continue  # Skip empty masks

            mean_vec = torch.sum(embedding * mask, dim=(1, 2)) / torch.sum(mask)
            mask_mean.append(mean_vec)

        if len(mask_mean) == 0:
            return 0.0  # No non-empty clusters, no loss

        mask_mean = torch.stack(mask_mean)

        # Variance loss
        var_loss = 0.0
        for cluster_id in range(num_clusters):
            mask = instance_mask[cluster_id].unsqueeze(0).float()

            # Skip if the mask is empty
            if torch.sum(mask) == 0:
                continue

            # Calculate variance
            if len(mask_mean) == 0:
              variance = torch.tensor(0.0, device=embedding.device)
            else:
              variance = torch.norm(embedding - mask_mean[cluster_id].unsqueeze(1).unsqueeze(2), dim=0)

            var_loss += torch.mean(torch.relu(variance * mask - self.delta_var) ** 2)

        # Divide by the number of non-empty clusters
        num_non_empty_clusters = len(mask_mean)
        var_loss = var_loss / num_non_empty_clusters if num_non_empty_clusters > 0 else torch.tensor(0.0, device=embedding.device)

        # Distance loss
        dist_loss = 0.0  # Initialize dist_loss to 0.0
        if num_clusters > 1:
            for i in range(num_clusters):
                for j in range(i + 1, num_clusters):
                    # Skip distance calculation if either cluster is empty
                    if torch.sum(instance_mask[i]) == 0 or torch.sum(instance_mask[j]) == 0:
                        continue

                    dist = torch.norm(mask_mean[i] - mask_mean[j], p=self.norm)
                    dist_loss += torch.relu(self.delta_dist - dist) ** 2
            # Divide by the correct number of distances
            dist_loss = dist_loss / (num_non_empty_clusters * (num_non_empty_clusters - 1) / 2) if num_non_empty_clusters > 1 else torch.tensor(0.0, device=embedding.device)
        # else: # Remove unnecessary else
        #     dist_loss = 0.0

        # Regularization loss
        reg_loss = torch.mean(torch.norm(mask_mean, p=self.norm, dim=1)) if num_non_empty_clusters > 0 else torch.tensor(0.0, device=embedding.device)

        return self.alpha * var_loss + self.beta * dist_loss + self.gamma * reg_loss