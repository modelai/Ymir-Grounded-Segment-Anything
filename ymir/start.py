import glob
import json
import logging
import random
import subprocess
import sys

from easydict import EasyDict as edict
from ymir_exc import monitor
from ymir_exc import result_writer as rw
from ymir_exc.util import get_bool, get_merged_config


def start() -> int:
    cfg = get_merged_config()

    if cfg.ymir.run_infer:
        _run_infer(cfg)

    return 0


def _run_infer(cfg: edict) -> None:
    monitor.write_monitor_logger(percent=0.0)
    logging.info(f"infer config: {cfg.param}")

    index_file = cfg.ymir.input.candidate_index_file
    text_prompt = cfg.param.text_prompt
    device = cfg.param.device or "cuda"
    box_threshold = float(cfg.param.box_threshold)
    text_threshold = float(cfg.param.text_threshold)
    sam_vit = cfg.param.sam_vit or "vit_b"
    assert sam_vit in ['vit_b', 'vit_l', 'vit_h']

    sam_checkpoint = glob.glob(f'ymir/sam_{sam_vit}*.pth')[0]

    logging.info(f'use device {device} to run task')

    command = [
        'python3', 'grounded_sam_demo.py', '--input', index_file, '--config',
        'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py', '--sam_checkpoint', sam_checkpoint,
        '--grounded_checkpoint', 'ymir/groundingdino_swint_ogc.pth', '--device', device, '--box_threshold',
        str(box_threshold), '--text_threshold',
        str(text_threshold), '--text_prompt', text_prompt
    ]

    logging.info(f'running command: {command}')
    subprocess.run(command, check=True)

    use_fake_label = get_bool(cfg, 'use_fake_label', False)
    if use_fake_label:
        class_names = cfg.param.class_names
        coco_results = modify_command_results(cfg.ymir.output.infer_result_file, class_names)
        rw.write_infer_result(infer_result=coco_results, algorithm='segmentation')
    else:
        # 会导致前端错误：没有指定类别
        class_names = [f'class_{i}' for i in range(16)]

    # if task done, write 100% percent log
    logging.info('infer done')
    monitor.write_monitor_logger(percent=1.0)


def modify_command_results(infer_result_file, class_names) -> dict:
    with open(infer_result_file, 'r') as fp:
        coco_results = json.load(fp)

    # use fake label
    for cat_info in coco_results['categories']:
        if len(class_names) == 0:
            cat_info['name'] = 'unknown'
        else:
            cat_info['name'] = random.choice(class_names)

    return coco_results


if __name__ == '__main__':
    sys.exit(start())
