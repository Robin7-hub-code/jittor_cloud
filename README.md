# Jittor 点云去噪系统（ModelNet40 + PCT）

本项目实现了一个可组合、模块化的点云去噪流程，覆盖：
- ModelNet40 数据下载与预处理
- 基于 Jittor 的 PCT 风格点云去噪模型
- 训练与评估（CD，EMD 可选）
- 去噪结果及残差数据导出
- 多视角可视化（matplotlib）

## 1. 环境准备

```bash
pip install -r requirements.txt
```

> 首次使用 Jittor 时，可能会自动编译算子，通常需要约 3~20 分钟（取决于 CPU/GPU 与网络）。如遇环境问题可参考 Jittor 官方文档：https://cg.cs.tsinghua.edu.cn/jittor/

## 2. 数据准备

### 2.1 下载 ModelNet40

```bash
python data/download_modelnet40.py --output-dir data/raw
```

### 2.2 预处理为训练/测试 split

```bash
python data/preprocess_modelnet40.py \
  --raw-dir data/raw/modelnet40_ply_hdf5_2048 \
  --output-dir data/processed \
  --num-points 1024
```

输出文件：
- `data/processed/train.npz`
- `data/processed/test.npz`

## 3. 训练

```bash
python train.py \
  --data-dir data/processed \
  --epochs 100 \
  --batch-size 16 \
  --noise-type gaussian \
  --noise-std 0.02 \
  --save-dir checkpoints
```

可切换噪声类型：
- `gaussian`
- `uniform`
- `mixed`

## 4. 评估与去噪

```bash
python eval.py \
  --data-dir data/processed \
  --checkpoint checkpoints/pct_denoiser_epoch_100.pkl \
  --batch-size 16 \
  --noise-type gaussian \
  --with-emd \
  --save-vis-dir outputs/vis_bundles \
  --save-metrics outputs/metrics.json
```

评估输出：
- `outputs/metrics.json`：整体 CD（及可选 EMD）统计
- `outputs/vis_bundles/*.npz`：单样本可视化数据包（clean/noisy/denoised/residual）

## 5. 单样本推理

```bash
python infer.py \
  --checkpoint checkpoints/pct_denoiser_epoch_100.pkl \
  --input your_points.npy \
  --output outputs/infer_result.npz
```

`--input` 支持：
- `.npy`（形状 `[N,3]` 或 `[B,N,3]`）
- `.npz`（默认读取 key=`points`，若不存在则读取第一个 key）

## 6. 可视化（多角度 + 残差）

```bash
python vis.py \
  --bundle outputs/vis_bundles/sample_0000_label_0.npz \
  --save-prefix outputs/compare/sample_0000 \
  --angles "30,45;30,135;60,45"
```

可选交互显示：
```bash
python vis.py \
  --bundle outputs/vis_bundles/sample_0000_label_0.npz \
  --interactive
```

## 7. 项目结构

```text
jittor_cloud/
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   ├── download_modelnet40.py
│   └── preprocess_modelnet40.py
├── models/
│   ├── __init__.py
│   └── pct_denoiser.py
├── utils/
│   ├── __init__.py
│   ├── io.py
│   ├── losses.py
│   ├── metrics.py
│   └── noise.py
├── train.py
├── eval.py
├── infer.py
├── vis.py
├── requirements.txt
└── README.md
```

## 8. 模块化与扩展建议

- 新模型：在 `models/` 新增网络并替换 `train.py/eval.py` 的模型实例化。
- 新噪声：在 `utils/noise.py` 扩展噪声函数即可复用训练与评估流程。
- 新指标：在 `utils/metrics.py` 新增函数并在 `eval.py` 统一汇总。
- 新可视化：可在 `vis.py` 中增加 Open3D 交互分支。
