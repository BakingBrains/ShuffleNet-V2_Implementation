import numpy as np
import os
from PIL import Image
import torch


def process_image(image_path):
    """Process an image path into a PyTorch tensor"""

    image = Image.open(image_path)
    # Resize
    img = image.resize((256, 256))

    # Center crop
    width = 256
    height = 256
    new_width = 224
    new_height = 224

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    img = img.crop((left, top, right, bottom))

    # Convert to numpy, transpose color dimension and normalize
    img = np.array(img).transpose((2, 0, 1)) / 256

    # Standardization
    means = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    stds = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

    img = img - means
    img = img / stds

    img_tensor = torch.Tensor(img)

    return img_tensor


class shufflenet_inference:
    def __init__(self, model_weight, classes):
        self.model_weight = torch.load(model_weight, map_location='cuda')
        self.classes = open(classes, 'r')
        self.categories = [line.split(',') for line in self.classes.readlines()]

    def inference_on_image(self, img_path):
        x = process_image(img_path)
        pred_labels = self.model_weight(torch.unsqueeze(x.cuda(), 0))
        predict_labels = np.array(pred_labels.cpu().detach().numpy()).ravel()
        index = list(predict_labels).index(max(list(predict_labels)))
        print(self.categories)
        #print("predicted labels is :", self.categories[index])
        return self.categories[index]

    def inference_on_folder(self, folder_path):
        for filename in os.listdir(folder_path):
            x = process_image(os.path.join(folder_path, filename))
            pred_labels = self.model_weight(torch.unsqueeze(x.cuda(), 0))
            # print(pred_labels)
            predict_labels = np.array(pred_labels.cpu().detach().numpy()).ravel()
            index = list(predict_labels).index(max(list(predict_labels)))
            #print("predicted labels is :", self.categories[index])
            return self.categories[index]


#Single_img = shufflenet_inference('/content/ShufflenetV2_model_4.pt', '/content/classes.txt').inference_on_image('/content/data/hymenoptera_data/train/bees/3006264892_30e9cced70.jpg')
#folder_ = shufflenet_inference('/content/ShufflenetV2_model_4.pt', '/content/classes.txt').inference_on_folder('/content/test')