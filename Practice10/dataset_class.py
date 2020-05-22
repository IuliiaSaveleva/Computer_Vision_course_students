import pickle
import os

from torch.utils.data import Dataset
from PIL import Image


class ValidationDataset(Dataset):
    def __init__(self, root, query_and_retrieval_pickle, transform=None):
        """Imports dataset from folder structure.

        Args:
            root: (string) Folder where the image samples are kept.
            transform: (Object) Image processing transformations.

        Attributes: 
            classes: (list) List of the class names.
            class_to_idx: (dict) pairs of (class_name, class_index).
            samples: (list) List of (sample_path, class_index) tuples.
            targets: (list) class_index value for each image in dataset.
        """

        super(ValidationDataset, self).__init__()
        # ORIGINAL STRATEGY - Couldn't customize labels to include broader
        # folder structures.
        # self.data = datasets.ImageFolder(image_path, transform)

        self.transform = transform
        with open(query_and_retrieval_pickle, 'rb') as f:
            dataset = pickle.load(f)
            joined_paths = dataset['query'] + dataset['retrieval']
        self.query_count = len(dataset['query'])
        self.classes, self.class_to_idx = self._find_classes(joined_paths)
        self.samples = self.make_dataset(root, joined_paths, self.class_to_idx)
        self.targets = [s[1] for s in self.samples]

    def _find_classes(self, paths):
        """Creates classes from the folder structure.

        Args:
            dir: (string) Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir),
            and class_to_idx is a dictionary.
        """

        classes = {}

        for path in paths:
            split = path.split('/')
            product_id = split[0]
            classes[product_id] = 0

        classes = sorted(list(classes.keys()))
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx

    def make_dataset(self, dir, paths, class_to_idx):
        """Returns a list of image path, and target index

        Args:
            dir: (string) The path of each image sample
            class_to_idx: (dict: string, int) Sorted classes, mapped to int

        Returns:
            images: (list of tuples) Path and mapped class for each sample
        """

        images = []

        for path in paths:
            split = path.split('/')
            product_id = split[0]
            item = (os.path.join(dir, path), class_to_idx[product_id])
            images.append(item)

        return images

    def get_class_dict(self):
        """Returns a dictionary of classes mapped to indicies."""
        return self.class_to_idx

    def __getitem__(self, index):
        """Returns tuple: (tensor, int) where target is class_index of
        target_class.

        Args:
            idx: (int) Index.
        """

        path, target = self.samples[index]
        sample = default_loader(path)
        sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return len(self.samples)


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)