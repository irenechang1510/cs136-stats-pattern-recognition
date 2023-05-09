import os
import random
from torch.utils.data import Dataset
from torchvision.io import read_image


def list_images(folder): 
    """
    This function list all `.jpg` images in a folder and return paths of these images in a list.
    args:
        folder: string, a path 
    returns: 
        a list of paths. For example, if an image `img0.jpg` is in the folder, then the path of 
        the image is `folder + "/img0.jpg"`

    """

    res = []
    for file in os.listdir(folder):
        # check only text files
        if file.endswith('.jpg'):
            res.append(os.path.join(folder, file))

    return res



class BeanImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
      # TODO: read in all image paths from `img_dir`. In this assignment, `img_dir` should be `data/train`, `data/validation`, or `data/test`
      angular = list_images(img_dir + "/angular_leaf_spot")
      rust = list_images(img_dir + "/bean_rust")
      healthy = list_images(img_dir + "/healthy")

      # TODO: according to the subfolder ("angular_leaf_spot", "bean_rust", or "healthy"), assign labels (0, 1, or 2) to these images 
      all_images = []
      all_labels = []
      
      all_images.extend(angular)
      all_labels.extend([0] * len(angular))

      all_images.extend(rust)
      all_labels.extend([1] * len(rust))

      all_images.extend(healthy)
      all_labels.extend([2] * len(healthy))

      # TODO: store all images and their labels to member variables 
      self.img_labels = all_labels
      self.img_dir = all_images
      self.transform = transform
      self.target_transform = target_transform

    def __len__(self):

        # TODO: get the size of the dataset and return it. 
        return len(self.img_labels)

    def __getitem__(self, idx):

        # TODO: locate the path and label of the `idx`-th image; read in the image to the **float** tensor `image`; and assign its label to `label`
        img_path = os.path.join(self.img_dir[idx])
        image = read_image(img_path)
        label = self.img_labels[idx]

        # apply necessary transformations if necessary
        # TODO: read "https://pytorch.org/vision/stable/transforms.html" to find more details

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
