# ImageLoss: Custom Image Similarity Loss Function for PyTorch

`ImageLoss` is a custom PyTorch loss function designed for image comparison tasks.  
It combines three components:
1. **Pixel-wise Huber loss**
2. **Spatial difference Huber loss (horizontal and vertical)**
3. **Channel-wise difference Huber loss**

This loss is particularly useful for tasks like **image restoration**, **denoising**, or **super-resolution**, where both local structure and channel continuity matter.

---

## Installation

No special installation is required. Just copy `ImageLoss` and the associated functions into your PyTorch project.

---

## Usage

```python
from image_loss import ImageLoss  # if saved as image_loss.py

x = torch.randn(8, 3, 32, 32)  # Predicted images
y = torch.randn(8, 3, 32, 32)  # Ground-truth images

criterion = ImageLoss(alpha=10, beta=5, delta=5/255)
loss = criterion(x, y).mean()
