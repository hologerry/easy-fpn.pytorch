import json
import os
import pickle
import random  # noqa
from typing import List, Tuple, Dict

import torch
import torch.utils.data.dataset
from PIL import Image, ImageOps  # noqa
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch import Tensor
# from torchvision.datasets import CocoDetection
from torchvision.datasets import VisionDataset
from tqdm import tqdm

from bbox import BBox
from dataset.base import Base
from io import StringIO
import sys


class ReceiptDetection(VisionDataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None):
        super(ReceiptDetection, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, path

    def __len__(self):
        return len(self.ids)


class Receipt(Base):

    class Annotation(object):
        class Object(object):
            def __init__(self, bbox: BBox, label: int):
                super().__init__()
                self.bbox = bbox
                self.label = label

            def __repr__(self) -> str:
                return 'Object[label={:d}, bbox={!s}]'.format(
                    self.label, self.bbox)

        def __init__(self, filename: str, objects: List[Object]):
            super().__init__()
            self.filename = filename
            self.objects = objects

    # Change the num_classes as well!!
    # Do not forget the background
    CATEGORY_TO_LABEL_DICT = {
        'background': 0, 'receipt': 1, 'Rotate0': 2, 'Rotate90': 3, 'Rotate180': 4, 'Rotate270': 5
    }

    LABEL_TO_CATEGORY_DICT = {v: k for k, v in CATEGORY_TO_LABEL_DICT.items()}

    def __init__(self, path_to_data_dir: str, mode: Base.Mode, image_min_side: float, image_max_side: float):
        super().__init__(path_to_data_dir, mode, image_min_side, image_max_side)

        path_to_coco_dir = os.path.join(self._path_to_data_dir, 'receipt_0158_0062_photo')
        path_to_annotations_dir = os.path.join(path_to_coco_dir, 'annotations')
        path_to_caches_dir = os.path.join('caches', 'receipt_0158_0062_photo', f'{self._mode.value}')
        path_to_image_ids_pickle = os.path.join(path_to_caches_dir, 'image-ids.pkl')
        path_to_image_id_dict_pickle = os.path.join(path_to_caches_dir, 'image-id-dict.pkl')
        path_to_image_id_to_file_name = os.path.join(path_to_caches_dir, 'image-id-to-file-name.pkl')
        path_to_image_ratios_pickle = os.path.join(path_to_caches_dir, 'image-ratios.pkl')

        if self._mode == Receipt.Mode.TRAIN:
            path_to_jpeg_images_dir = os.path.join(path_to_coco_dir, 'images_train')
            path_to_annotation = os.path.join(path_to_annotations_dir, 'train.json')
        elif self._mode == Receipt.Mode.EVAL:
            path_to_jpeg_images_dir = os.path.join(path_to_coco_dir, 'images_val')
            path_to_annotation = os.path.join(path_to_annotations_dir, 'val.json')
        else:
            raise ValueError('invalid mode')

        coco_dataset = ReceiptDetection(root=path_to_jpeg_images_dir, annFile=path_to_annotation)

        if os.path.exists(path_to_image_ids_pickle) and os.path.exists(path_to_image_id_dict_pickle):
            print('loading cache files...')

            with open(path_to_image_ids_pickle, 'rb') as f:
                self._image_ids = pickle.load(f)

            with open(path_to_image_id_dict_pickle, 'rb') as f:
                self._image_id_to_annotation_dict = pickle.load(f)

            with open(path_to_image_id_to_file_name, 'rb') as f:
                self._image_id_to_file_name = pickle.load(f)

            with open(path_to_image_ratios_pickle, 'rb') as f:
                self._image_ratios = pickle.load(f)
        else:
            print('generating cache files...')

            os.makedirs(path_to_caches_dir, exist_ok=True)

            self._image_ids: List[str] = []
            self._image_id_to_annotation_dict: Dict[str, Receipt.Annotation] = {}
            self._image_id_to_file_name: Dict[str, str] = {}
            self._image_ratios = []

            for idx, (image, annotation, path) in enumerate(tqdm(coco_dataset)):
                if len(annotation) > 0:
                    image_id = str(annotation[0]['image_id'])  # all image_id in annotation are the same
                    self._image_ids.append(image_id)
                    self._image_id_to_file_name[image_id] = path
                    self._image_id_to_annotation_dict[image_id] = Receipt.Annotation(
                        # filename=os.path.join(path_to_jpeg_images_dir, '{:012d}.jpg'.format(int(image_id))),
                        filename=os.path.join(path_to_jpeg_images_dir, path),
                        objects=[Receipt.Annotation.Object(
                            bbox=BBox(  # `ann['bbox']` is in the format [left, top, width, height]
                                left=ann['bbox'][0],
                                top=ann['bbox'][1],
                                right=ann['bbox'][0] + ann['bbox'][2],
                                bottom=ann['bbox'][1] + ann['bbox'][3]
                            ),
                            label=ann['category_id'])
                            for ann in annotation]
                    )

                    ratio = float(image.width / image.height)
                    self._image_ratios.append(ratio)

            with open(path_to_image_ids_pickle, 'wb') as f:
                pickle.dump(self._image_ids, f)

            with open(path_to_image_id_dict_pickle, 'wb') as f:
                pickle.dump(self._image_id_to_annotation_dict, f)

            with open(path_to_image_id_to_file_name, 'wb') as f:
                pickle.dump(self._image_id_to_file_name, f)

            with open(path_to_image_ratios_pickle, 'wb') as f:
                pickle.dump(self.image_ratios, f)

    def __len__(self) -> int:
        return len(self._image_id_to_annotation_dict)

    def __getitem__(self, index: int) -> Tuple[str, Tensor, Tensor, Tensor, Tensor]:
        image_id = self._image_ids[index]
        annotation = self._image_id_to_annotation_dict[image_id]

        bboxes = [obj.bbox.tolist() for obj in annotation.objects]
        labels = [obj.label for obj in annotation.objects]

        bboxes = torch.tensor(bboxes, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)

        image = Image.open(annotation.filename).convert('RGB')  # for some grayscale images

        # random flip on only training mode
        # Do not flip on receipt data
        # if self._mode == Receipt.Mode.TRAIN and random.random() > 0.5:
        #     image = ImageOps.mirror(image)
        #     bboxes[:, [0, 2]] = image.width - bboxes[:, [2, 0]]  # index 0 and 2 represent `left` and `right` respectively

        image, scale = Receipt.preprocess(image, self._image_min_side, self._image_max_side)
        scale = torch.tensor(scale, dtype=torch.float)
        bboxes *= scale

        return image_id, image, scale, bboxes, labels

    def evaluate(self, path_to_results_dir: str, image_ids: List[str], bboxes: List[List[float]], classes: List[int], probs: List[float]) -> Tuple[float, str]:
        self._write_results(path_to_results_dir, image_ids, bboxes, classes, probs)

        annType = 'bbox'
        path_to_coco_dir = os.path.join(self._path_to_data_dir, 'receipt_0158_0062_photo')
        path_to_annotations_dir = os.path.join(path_to_coco_dir, 'annotations')
        path_to_annotation = os.path.join(path_to_annotations_dir, 'val.json')

        cocoGt = COCO(path_to_annotation)
        cocoDt = cocoGt.loadRes(os.path.join(path_to_results_dir, 'results.json'))

        cocoEval = COCOeval(cocoGt, cocoDt, annType)
        cocoEval.evaluate()
        cocoEval.accumulate()

        original_stdout = sys.stdout
        string_stdout = StringIO()
        sys.stdout = string_stdout
        cocoEval.summarize()
        sys.stdout = original_stdout

        mean_ap = cocoEval.stats[0].item()  # stats[0] records AP@[0.5:0.95]
        detail = string_stdout.getvalue()

        return mean_ap, detail

    def _write_results(self, path_to_results_dir: str, image_ids: List[str], bboxes: List[List[float]], classes: List[int], probs: List[float]):
        results = []
        for image_id, bbox, cls, prob in zip(image_ids, bboxes, classes, probs):
            results.append(
                {
                    'image_id': int(image_id),  # COCO evaluation requires `image_id` to be type `int`
                    'file_name': self._image_id_to_file_name[image_id],
                    'category_id': cls,
                    'bbox': [   # format [left, top, width, height] is expected
                        bbox[0],
                        bbox[1],
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1]
                    ],
                    'score': prob
                }
            )

        with open(os.path.join(path_to_results_dir, 'results.json'), 'w') as f:
            json.dump(results, f)

    @property
    def image_ratios(self) -> List[float]:
        return self._image_ratios

    @staticmethod
    def num_classes() -> int:
        return 6
