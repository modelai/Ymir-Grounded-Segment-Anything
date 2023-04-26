# YMIR-GroundingSAM

## 启动命令
```
python ymir/start.py

python grounded_sam_demo.py \
    --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --grounded_checkpoint ymir/groundingdino_swint_ogc.pth \
    --sam_checkpoint ymir/sam_vit_b_01ec64.pth \
    --input ymir/demo/000019.jpg \
    --text_prompt cat \
    --output_dir outputs \
    --device cuda
```

## 制作镜像

采用双阶段构建可以有效减少镜像大小，若采用 devel 镜像直接构建约20G, 采用 runtime 镜像约10G。

- 预备工作，将需要用到的权重下载到指定目录

```
ymir/groundingdino_swint_ogc.pth
ymir/sam_vit_b_01ec64.pth
ymir/bert-base-uncased
├── config.json
├── pytorch_model.bin
├── special_tokens_map.json
├── tokenizer_config.json
└── vocab.txt
```

- 先用 `pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel` 编译生成 deformable-detr 模块

    - `GroundingDINO/groundingdino/_C.cpython-38-x86_64-linux-gnu.so`

- 再用 `pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime` 构建最终镜像

```
docker build -t youdaoyzbx/ymir-executor:ymir2.1.0-grounding-sam-cu111-infer -f ymir/dockerfile .
```
