from torch import nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, **kwargs):
        """
        Forward pass of the model.
        This forward function should return a dict with a 'loss' key.

        Returns:
            {
                'loss': torch.Tensor,
                'x': torch.Tensor,  # Optional, can be used for validation or testing
            }
        """
        ...

    def get_all_params(self):
        """
        Get all parameters of the model.
        You can override this method to return parameter groups.
        For example:
            return [
                {
                    'params': self.layer1.parameters(),
                    'lr': 5e-4,
                    'weight_decay': 0.0
                },
                {
                    'params': self.layer2.parameters(),
                    'lr': 1e-4,
                    'weight_decay': 0.5
                }
            ]
        By default, this method returns all parameters of the model.

        Returns:
            list: List of all model parameters.
        """
        return list(self.parameters())
