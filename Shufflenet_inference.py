import numpy as np
from PIL import Image
import torch
import torchvision.models as models


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

class ShuffleNet_v2:
    def __init__ (self,classes, img_path):
        self.model = models.squeezenet1_0(pretrained=True)
        self.model=self.model.cuda()
        self.img_path = img_path
        self.crimefile = open(classes, 'r')
        self.categories = [line.split(',') for line in self.crimefile.readlines()]

    def predict(self):
        x = process_image(self.img_path)
        pred_labels = self.model(torch.unsqueeze(x.cuda(), 0))
        # print(pred_labels)
        print(pred_labels)
        if len(pred_labels)>0:
            predict_labels = np.array(pred_labels.cpu().detach().numpy()).ravel()
            index = list(predict_labels).index(max(list(predict_labels)))
            #print("predicted labels is :", self.categories[index])
            classname = self.categories[index].pop()
            json_obj = {"classname": classname}
            return json_obj
        else:
            print("no label detected")

# a=ShuffleNet_v2(r'C:/Users/Prince_Shaks/PycharmProjects/Shufflenet/imagenet_classes.txt', 'C:/Users/Prince_Shaks/PycharmProjects/Shufflenet/test/ant.jpg').predict()
# print(a)