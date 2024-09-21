# Training Pipeline

The training process is designed to minimize the L1 loss between the predicted and ground truth keypoints. The optimizer used is Adam, which adjusts the learning rate during training to achieve better convergence.

## Key Components

1. **Loss Function**: L1 Loss is used to calculate the difference between the predicted keypoints and the ground truth.

   ```python
   criterion = nn.L1Loss()
   ```

2. **Optimizer**: Adam optimizer is employed to update the model parameters during training.

   ```python
   optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
   ```

3. **Training Loop**:

   - For each batch, a forward pass is computed, the loss is calculated, and gradients are backpropagated.
   - The optimizer then updates the model's weights accordingly.

   ```python
   def train_batch(imgs, kps, model, criterion, optimizer):
       model.train()
       optimizer.zero_grad()

       # Forward pass
       kps_pred = model(imgs)
       loss = criterion(kps_pred, kps)

       # Backward pass
       loss.backward()
       optimizer.step()

       return loss
   ```

4. **Evaluation**: The model's performance is evaluated on the test dataset by computing the same loss metric. The evaluation process is done without updating the model's weights.

   ```python
   @torch.no_grad()
   def test_batch(imgs, kps, model, criterion):
       model.eval()

       # Forward pass
       kps_pred = model(imgs)
       loss = criterion(kps_pred, kps)

       return loss
   ```

### Training Example:

```python
for epoch in range(num_epochs):
    for imgs, kps in train_loader:
        loss = train_batch(imgs, kps, model, criterion, optimizer)
        # Log training loss

    # Perform evaluation on the test set
    for imgs, kps in test_loader:
        val_loss = test_batch(imgs, kps, model, criterion)
        # Log test loss
```
