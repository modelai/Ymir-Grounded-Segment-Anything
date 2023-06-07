# YMIR-GroundingSAM

结合YMIR, Grounding-DINO 与 Segment Everything Model 的实例分割镜像，仅支持推理功能。

- 输入：文本 + 图像 (不支持中文)

- 输出：文件对应目标的分割结果

## 仓库地址

- 🌈代码：[modelai/Ymir-Grounded-Segment-Anything](https://github.com/modelai/Ymir-Grounded-Segment-Anything)

- 📀镜像: youdaoyzbx/ymir-executor:ymir2.5.0-grounded-huge-sam-cu111-infer

- 📘文档：[ymir-executor-doc](https://ymir-executor-fork.readthedocs.io/zh/latest/)

- 🏠基地：[ymir-executor-fork](https://github.com/modelai/ymir-executor-fork)

## 推理参数

| 超参数 | 默认值 | 类型 | 说明 | 建议 |
| - | - | - | - | - |
| hyper-parameter | default value | type | note | advice |
| text_prompt | objects | 字符串 | grounding-dino 的 prompt | - |
| device | cuda | cuda/cpu | 推理设备 | 有gpu 采用 cuda, 无gpu 采用 cpu |
| box_threshold | 0.3 | 浮点数 | 置信度阈值 | - |
| text_threshold | 0.25 | 浮点数 | 文本特征阈值 | - |
| sam_vit | vit_b | vit_h/l/b | huge/large/base | vit_h 精度最高, vit_b 最快 |

## 启动命令

修改后的仅能在ymir环境下运行

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
docker build -t youdaoyzbx/ymir-executor:ymir2.5.0-grounded-huge-sam-cu111-infer -f ymir/dockerfile .
```
