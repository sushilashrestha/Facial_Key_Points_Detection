````markdown
# Training Pipeline

The training process is designed to minimize the L1 loss between the predicted and ground truth keypoints. The optimizer used is Adam, which adjusts the learning rate during training for better convergence.

### Key Components

1.  **Loss Function**: L1 Loss is used to calculate the difference between the predicted keypoints and the ground truth.

        ```python
        criterion = nn.L1Loss()
        ```

2.  **Optimizer**: Adam optimizer is employed to update the model parameters.

        ```python
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        ```
````
