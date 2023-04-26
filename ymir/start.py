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
    device = cfg.param.device
    box_threshold = float(cfg.param.box_threshold)
    text_threshold = float(cfg.param.text_threshold)

    logging.info(f'use device {device} to run task')

    command = [
        'python3', 'grounded_sam_demo.py', '--input', index_file, '--config',
        'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py', '--sam_checkpoint',
        'ymir/sam_vit_b_01ec64.pth', '--grounded_checkpoint', 'ymir/groundingdino_swint_ogc.pth', '--device', device,
        '--box_threshold',
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
        cat_info['name'] = random.choice(class_names)

    return coco_results


if __name__ == '__main__':
    sys.exit(start())
