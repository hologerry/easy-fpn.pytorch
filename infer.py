import argparse
import os

import torch
from PIL import ImageDraw
from torchvision.transforms import transforms

from backbone.base import Base as BackboneBase
from bbox import BBox
from config.eval_config import EvalConfig as Config
from dataset.base import Base as DatasetBase
from model import Model
from roi.pooler import Pooler


def _infer(path_to_input_image: str, path_to_output_image: str, path_to_checkpoint: str,
           dataset_name: str, backbone_name: str, prob_thresh: float):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = BackboneBase.from_name(backbone_name)(pretrained=False)
    dataset_class = DatasetBase.from_name(dataset_name)
    model = Model(backbone, dataset_class.num_classes(), pooler_mode=Config.POOLER_MODE,
                  anchor_ratios=Config.ANCHOR_RATIOS, anchor_scales=Config.ANCHOR_SCALES,
                  rpn_pre_nms_top_n=Config.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=Config.RPN_POST_NMS_TOP_N).to(device)
    model.load(path_to_checkpoint)

    image = transforms.Image.open(path_to_input_image)
    image_tensor, scale = dataset_class.preprocess(image, Config.IMAGE_MIN_SIDE, Config.IMAGE_MAX_SIDE)

    forward_input = Model.ForwardInput.Eval(image_tensor.to(device))
    forward_output: Model.ForwardOutput.Eval = model.eval().forward(forward_input)

    detection_bboxes = forward_output.detection_bboxes / scale
    detection_classes = forward_output.detection_classes
    detection_probs = forward_output.detection_probs

    kept_indices = detection_probs > prob_thresh
    detection_bboxes = detection_bboxes[kept_indices]
    detection_classes = detection_classes[kept_indices]
    detection_probs = detection_probs[kept_indices]

    draw = ImageDraw.Draw(image)

    for bbox, cls, prob in zip(detection_bboxes.tolist(), detection_classes.tolist(), detection_probs.tolist()):
        color = ['red', 'green', 'blue', 'yellow', 'purple', 'white'][cls]
        bbox = BBox(left=bbox[0], top=bbox[1], right=bbox[2], bottom=bbox[3])
        category = dataset_class.LABEL_TO_CATEGORY_DICT[cls]

        draw.rectangle(((bbox.left, bbox.top), (bbox.right, bbox.bottom)), outline=color)
        draw.text((bbox.left, bbox.top), text=f'{category:s} {prob:.3f}', fill=color)

    image.save(path_to_output_image)
    print(f'Output image is saved to {path_to_output_image}')


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('input', type=str, help='path to input image')
        parser.add_argument('output', type=str, help='path to output result image')
        parser.add_argument('-c', '--checkpoint', type=str, required=True, help='path to checkpoint')
        parser.add_argument('-s', '--dataset', type=str, choices=DatasetBase.OPTIONS, required=True, help='name of dataset')
        parser.add_argument('-b', '--backbone', type=str, choices=BackboneBase.OPTIONS, required=True, help='name of backbone model')
        parser.add_argument('-p', '--probability_threshold', type=float, default=0.6, help='threshold of detection probability')
        parser.add_argument('--image_min_side', type=float, help='default: {:g}'.format(Config.IMAGE_MIN_SIDE))
        parser.add_argument('--image_max_side', type=float, help='default: {:g}'.format(Config.IMAGE_MAX_SIDE))
        parser.add_argument('--anchor_ratios', type=str, help='default: "{!s}"'.format(Config.ANCHOR_RATIOS))
        parser.add_argument('--anchor_scales', type=str, help='default: "{!s}"'.format(Config.ANCHOR_SCALES))
        parser.add_argument('--pooler_mode', type=str, choices=Pooler.OPTIONS, help='default: {.value:s}'.format(Config.POOLER_MODE))
        parser.add_argument('--rpn_pre_nms_top_n', type=int, help='default: {:d}'.format(Config.RPN_PRE_NMS_TOP_N))
        parser.add_argument('--rpn_post_nms_top_n', type=int, help='default: {:d}'.format(Config.RPN_POST_NMS_TOP_N))
        args = parser.parse_args()

        path_to_input_image = args.input
        path_to_output_image = args.output
        path_to_checkpoint = args.checkpoint
        dataset_name = args.dataset
        backbone_name = args.backbone
        prob_thresh = args.probability_threshold

        os.makedirs(os.path.join(os.path.curdir, os.path.dirname(path_to_output_image)), exist_ok=True)

        Config.setup(image_min_side=args.image_min_side, image_max_side=args.image_max_side,
                     anchor_ratios=args.anchor_ratios, anchor_scales=args.anchor_scales, pooler_mode=args.pooler_mode,
                     rpn_pre_nms_top_n=args.rpn_pre_nms_top_n, rpn_post_nms_top_n=args.rpn_post_nms_top_n)

        print('Arguments:')
        for k, v in vars(args).items():
            print(f'\t{k} = {v}')
        print(Config.describe())

        _infer(path_to_input_image, path_to_output_image, path_to_checkpoint, dataset_name, backbone_name, prob_thresh)

    main()
