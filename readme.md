# Facial Keypoint Detection

This project builds a deep learning model to detect facial keypoints from images. The model is based on a modified VGG16 architecture and is trained using PyTorch. Facial keypoint detection is crucial for applications such as face tracking, emotion recognition, and head pose estimation.

## Project Architecture

The project is divided into the following key components:

1. **Data Loading & Preprocessing**:

   - Images and keypoint annotations are loaded.
   - Images are normalized and resized to a standard size.

2. **Model Building**:

   - A pre-trained VGG16 model is fine-tuned for the task.
   - The final classifier layer outputs 136 coordinates (68 keypoints).

3. **Training Pipeline**:

   - The model is trained using the Adam optimizer and L1 loss function.
   - The training process includes periodic evaluation on a test dataset to monitor the model's performance.

4. **Result Visualization**:
   - Loss curves (training and validation) are plotted.
   - Predicted keypoints are visualized over the original images.

## Installation

### Requirements

- Python 3.8+
- PyTorch
- Matplotlib
- NumPy

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/facial-keypoint-detection.git
   cd facial-keypoint-detection
   ```
