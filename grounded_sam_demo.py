import argparse
import copy
import json
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_util
from tqdm import tqdm
from ymir_exc import result_writer as rw
from ymir_exc.monitor import write_monitor_logger

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import (clean_state_dict, get_phrases_from_posmap)
# segment anything
from segment_anything import SamPredictor, sam_model_registry


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    scores = []
    for logit in logits_filt:
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        score = round(logit.max().item(), 4)
        scores.append(score)
        pred_phrases.append(pred_phrase)

    return boxes_filt, scores, pred_phrases


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{'value': value, 'label': 'background'}]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1]  # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument("--grounded_checkpoint", type=str, required=True, help="path to checkpoint file")
    parser.add_argument("--sam_checkpoint", type=str, required=True, help="path to checkpoint file")
    parser.add_argument("--input", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    # parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=True, help="output directory")

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint

    if args.input.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        image_list = [args.input]
    elif args.input.lower().endswith(('.txt', '.tsv')):
        with open(args.input, 'r') as fp:
            image_list = [line.strip() for line in fp.readlines()]
    else:
        assert False, f'unknown input format {args.inpu}'

    text_prompt = args.text_prompt
    # output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device

    # make dir
    # os.makedirs(output_dir, exist_ok=True)

    # load grounding-dino model
    model = load_model(config_file, grounded_checkpoint, device=device)
    # initialize SAM
    sam_model_type = os.path.basename(sam_checkpoint)[4:9]
    build_sam = sam_model_registry[sam_model_type]
    predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

    N = len(image_list)

    coco_results = {}
    cat_id, img_id, ann_id = 0, 0, 1
    coco_results['images'] = []
    coco_results['annotations'] = []
    coco_results['categories'] = []
    label2id = {}
    for idx, image_path in enumerate(tqdm(image_list)):
        # load image
        image_pil, image = load_image(image_path)
        # visualize raw image
        # image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

        # run grounding dino model
        boxes_filt, scores, pred_phrases = get_grounding_output(model,
                                                                image,
                                                                text_prompt,
                                                                box_threshold,
                                                                text_threshold,
                                                                device=device,
                                                                with_logits=False)

        write_monitor_logger(round(idx / N, 2))
        basename = os.path.basename(image_path)

        size = image_pil.size
        H, W = size[1], size[0]
        img_info = dict(id=img_id, file_name=basename, width=W, height=H)
        coco_results['images'].append(img_info)

        if boxes_filt.size(0) == 0:
            img_id += 1
            continue

        predictor.set_image(np.array(image_pil))
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(device),
            multimask_output=False,
        )

        for box, mask, score, label in zip(boxes_filt.data.cpu().numpy(),
                                           masks.data.cpu().numpy(), scores, pred_phrases):
            label = label or 'unknown'

            if label not in label2id:
                label2id[label] = cat_id
                cat_info = dict(id=cat_id, name=str(label), supercategory="none")
                coco_results['categories'].append(cat_info)
                cat_id += 1

            x1, y1, x2, y2 = [float(zz) for zz in box]
            w, h = x2 - x1, y2 - y1

            # masks: B,C,H,W
            # mask: C, H, W
            rle_mask = mask_util.encode(np.asfortranarray(mask[0].astype(np.uint8)))
            area = float(mask_util.area(rle_mask))
            if area > 0:
                segmentation = rle_mask
                segmentation['counts'] = segmentation['counts'].decode('utf-8')
                ann_info = dict(id=ann_id,
                                image_id=img_id,
                                category_id=label2id[label],
                                confidence=float(score),
                                bbox=[x1, y1, w, h],
                                is_crowd=1,
                                area=area,
                                segmentation=segmentation)

                coco_results['annotations'].append(ann_info)
                ann_id += 1

        img_id += 1

    rw.write_infer_result(infer_result=coco_results, algorithm='segmentation')

    # draw output image
    # plt.figure(figsize=(10, 10))
    # plt.imshow(image)
    # for mask in masks:
    #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    # for box, label in zip(boxes_filt, pred_phrases):
    #     show_box(box.numpy(), plt.gca(), label)

    # plt.axis('off')
    # plt.savefig(os.path.join(output_dir, "grounded_sam_output.jpg"), bbox_inches="tight", dpi=300, pad_inches=0.0)

    # save_mask_data(output_dir, masks, boxes_filt, pred_phrases)
