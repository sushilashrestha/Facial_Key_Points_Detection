from PIL import Image
from matplotlib import pyplot as plt

from src.facial_key_point.utils.facial_key_point_detection import FacialKeyPointDetection

if __name__ == "__main__":
    image = Image.open('face.jpg').convert('RGB')
    facial_key_point_detection = FacialKeyPointDetection()
    image, kp = facial_key_point_detection.predict(image)

    plt.figure()
    plt.imshow(image)
    plt.scatterplot(kp[0], kp[1], s=4, c='r')
    plt.savefig('viz.png')