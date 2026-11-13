# macOS（Apple Silicon）运行说明

本仓库原为 Windows 便携版。macOS 分支保留同一套 Flask 网页界面和 FlowHMR 推理管线，
把 Win32 桌面壳、CUDA 运行环境和 CUDA 专用算子替换为 PyTorch MPS / CPU 路径。

- 适用：Apple Silicon（M1 及以上），macOS 14 或更新；Intel Mac 可以 `--device cpu` 运行，但会很慢。
- 已在本机验证：M1 Pro / 32 GB / macOS 26.5.1 / Python 3.12 / torch 2.8.0（MPS）。
  `check_install.py` 自检通过，命令行与网页两条入口均跑通完整流程（静态机位与 VGGT 机位各一次）。
- 没有 macOS 安装包；源码运行，界面使用浏览器打开。

## 一键安装

```bash
cd FlowHMR-Windows
PYTHON=/opt/homebrew/bin/python3.12 tools/macos/setup.sh
# 网络访问 PyPI 较慢时可换镜像：
PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ tools/macos/setup.sh
```

`setup.sh` 会初始化子模块并打上 `third_party/patches/0005-0006` 的设备补丁，把 Meta 的
DINOv3 hub 代码取到 `third_party/dinov3`（可选，失败时回退到运行时从 GitHub 拉取），
创建 `.venv`、安装 `requirements-macos.txt`、以可编辑模式安装 `third_party/YOLOX`，
然后下载模型并执行自检。需要 Python 3.10–3.12（推荐 3.12），系统需要 `ffmpeg`
（`brew install ffmpeg`；没有时自动使用 `imageio-ffmpeg`）。

模型约 10 GB，全部可断点续传。也可以分开执行：

```bash
.venv/bin/python tools/macos/download_models.py                 # 全部
.venv/bin/python tools/macos/download_models.py --skip vggt     # 先跳过运动相机模型
.venv/bin/python tools/macos/check_install.py                   # 自检
```

## 启动

```bash
tools/macos/run_web.sh                 # 默认 8080 端口，自动打开浏览器
FLOWHMR_PORT=8090 tools/macos/run_web.sh
tools/macos/run_web.sh --device cpu    # 强制 CPU
```

命令行批量处理：

```bash
.venv/bin/python run_v2m_demo_generation.py assets/example/example.mp4 --static_camera
```

`.venv/bin/python app.py --help` 与网页界面参数一致。

## 模型

`download_models.py` 从以下来源取文件，并沿用仓库内已有的 SHA-256 校验：

| 模型 | 来源 |
| --- | --- |
| FlowHMR 0.46B（1.8 GB） | hf-mirror `fafsaf1/flowhmr-0.46B` |
| YOLOX-L（434 MB） | GitHub Release，经 `ghfast.top` 镜像 |
| SMPL neutral J regressor | jsDelivr 上的 GVHMR 仓库 |
| SAM-3D-Body DINOv3（2.8 GB） | ModelScope `facebook/sam-3d-body-dinov3` |
| VGGT-Omega 1B（4.6 GB） | ModelScope `facebook/VGGT-Omega` |
| SMPL-H / MANO（需登录的包） | hf-mirror `lithiumice/models_hub/4_SMPLhub` |

`4_SMPLhub` 的对应关系：

| 镜像文件 | 本地路径 |
| --- | --- |
| `4_SMPLhub/smplh/neutral/model.npz` | `models/body_models/smplh/neutral/model.npz` |
| `4_SMPLhub/SMPLH/X_pkl/SMPLH_male.pkl` | `models/mano_v1_2/models/SMPLH_male.pkl` |
| `4_SMPLhub/SMPLH/X_pkl/SMPLH_female.pkl` | `models/mano_v1_2/models/SMPLH_female.pkl` |
| `4_SMPLhub/MANO/pkl/MANO_LEFT.pkl` | `models/mano_v1_2/models/MANO_LEFT.pkl` |
| `4_SMPLhub/MANO/pkl/MANO_RIGHT.pkl` | `models/mano_v1_2/models/MANO_RIGHT.pkl` |

已经手工下载过这些文件时，按上表放置后运行 `check_install.py` 即可，无需重新下载。
网页界面的「模型」页同样可以下载公开模型。

## macOS 适配内容

- 设备选择：`cuda` → `mps` → `cpu` 自动回退；Kernel 仍是上游的 FlowHMR 0.46B + SAM-3D-Body。
- YOLOX 始终在 CPU 上推理（`torchvision.ops.nms` 在 MPS 上没有可靠实现），检测结果随后交给 GPU 阶段。
- SAM-3D-Body 中写死的 `.cuda()` / `"cuda"` 改为模型自身设备；MPS 上关闭 `use_gpu_preprocess`，
  预处理回到 OpenCV 路径；无 BF16 时骨干转 FP16。
- DINOv3 骨干代码由 `setup.sh` 取到 `third_party/dinov3`，推理时不再访问 GitHub
  （`torch.hub` 在线拉取在国内网络会超时）；目录缺失时回退到 `torch.hub` 在线加载。
- MHR 的 TorchScript 正演使用 float64，MPS 无法执行：子模型固定留在 CPU，输入/输出在头部边界搬运。
- `get_module_device()` 原来把所有非 CUDA 设备当作 CPU，已改为读取参数真实设备。
- 注意力已使用数学后端（`SDPBackend.MATH`），不依赖 FlashAttention；RoPE 使用实数路径，避开 MPS 缺失的复数算子。
- 相机默认固定机位（不加载 VGGT）；选择运动相机时 VGGT 也能在 MPS/CPU 上运行，但明显更慢。
- 网页界面用浏览器访问；Windows 的 `FlowHMR.exe` 桌面壳、PowerShell 启动脚本和 CUDA 运行环境下载在 macOS 不参与运行。
- 子模块的两处改动以补丁形式放在 `third_party/patches/`，由 `setup.sh` 应用：
  `0005-sam3d-body-device-agnostic.patch`（DINOv3/MHR 的设备处理）与
  `0006-yolox-source-build-any-platform.patch`（只在 Linux 编译 C++ 算子）。

## 已知限制

- **HaMeR 手指精修不可用**：上游实现只接受 CUDA 设备，macOS 上会被自动跳过。
- **FBX 导出不可用**：`fbxsdkpy` 只有 Windows 轮子。NPZ、VMD、VRMA 导出正常。
- **接触修正（postprocess）默认关闭**：需要本地编译 `third_party/KimodoMotionCorrection`（CMake + Eigen）
  和 `third_party/RGBMocapCorrection`；没有编译时网页会隐藏该选项，原始模型输出仍可用。
- 实测速度（M1 Pro / 32 GB，1080p、30 fps）：YOLOX 检测约 30 帧/秒，SAM-3D-Body 特征约
  0.25–0.36 帧/秒（90 帧约 4–6 分钟），VGGT 采样帧约 1 秒/帧，FlowHMR 生成 90 帧约 5 秒。
  静态机位跳过 VGGT；SAM 批次自适应调节（首次运行从 1 帧起步，逐步升到 8–13 帧）。
- 内存与显存共享 32 GB；长视频建议缩短帧范围或降低 SAM 批次，界面里可调。
- 首次运行会加载模型，耗时会明显长于后续运行；特征缓存位于 `output/<视频名>/`。

### SAM 阶段耗时拆分

16 帧、批次 8（M1 Pro，合计约 39 秒）的分段计时：

| 段 | 占比 | 说明 |
| --- | --- | --- |
| DINOv3 ViT-H+ 骨干 | ~87% | 每张 512×512 裁剪图约 0.7 秒；身体批次 8 张、手部批次 16 张（左右手合并） |
| MHR 头（CPU float64） | ~5% | 6 个 decoder 层各算一次中间预测，共 12 次/批 |
| body / hand decoder | ~6% | 6 层 promptable decoder |
| prompt encoder | ~4% | 每批一次 dummy prompt |

因此 macOS 上的瓶颈是骨干网络的前向次数，而不是解码或 MHR。手部两路解码会额外跑
**两倍于帧数**的骨干前向；把手部采样间隔调到 4（界面「SAM 手部采样」或 `--sam_hand_stride 4`）
实测把同一段 38.6 秒降到 21.4 秒（约 1.8 倍），代价是手部 token 由邻近采样帧线性插值。

需要重新测量时设置环境变量：

```bash
FLOWHMR_SAM_PROFILE=1 tools/macos/run_web.sh
```

`SamPhaseProfiler` 会按模块列出 self/incl 耗时；它会在每段前后同步 MPS，因此总时间会被拉长约
20%，看比例即可。

### 为什么不把 SAM 整体转 FP16

- 骨干权重在 MPS 上已经是 FP16（无 BF16 时的回退），实测已经是较快的那个：整段 FP16 骨干
  38.6 秒 vs FP32 40.6 秒，只有约 5%。
- 对整个前向套 `torch.autocast("mps", float16)` 反而更慢：43.0 秒 vs 38.6 秒。
- 微基准（MPS，M1 Pro）：矩阵乘 FP16 20.5 ms / FP32 20.9 ms（几乎相同），
  16×16 卷积 11.9 / 30.7 ms，SDPA 注意力 41.9 / 46.3 ms。MPS 的 GEMM 与注意力没有真正的
  FP16 加速路径，只有卷积受益，而骨干里卷积只占一小部分。

结论：保持「骨干 FP16、其余 FP32」现状，不要追求全模型 FP16；要提速请减少骨干前向次数
（手部 stride、帧范围、批次）或改用下面的 MLX 骨干。


### 可选：MLX 骨干

SAM 的 DINOv3 ViT-H+ 骨干有一段 MLX 实现（`flowhmr/utils/mlx_dinov3.py`），权重直接从已加载的
torch 骨干转换，移植了 32 层全局注意力、轴向 RoPE、SwiGLU、LayerScale 和 4 个 storage token。
解码器、MHR 头仍留在 torch 里。默认不启用。

```bash
.venv/bin/pip install -r requirements-macos-mlx.txt    # mlx>=0.29，仅 macOS
```

启用方式二选一：界面「SAM 骨干运行时」选 **MLX（macOS）**，或命令行
`--sam_backbone mlx`。没有安装 MLX 时界面会把该选项置灰，接口也会直接报错。

实测（M1 Pro，8 帧、批次 8，同进程交替两轮）：

| 指标 | torch 骨干 | MLX 骨干 |
| --- | --- | --- |
| 整个 SAM 阶段 | 22.9 / 22.4 s | 18.4 / 18.1 s（**1.24 倍**） |
| token 相对 L2（MLX FP16 vs torch FP16） | — | 3.3e-3（0.33%） |
| 切回 torch 后与切之前对比 | — | **max\|d\| = 0**（权重没有离开进程，逐位一致） |
| 权重转换耗时 | — | 0.6–0.7 s |
| wired 峰值 / 进程退出后 | 4.5 GB | 6.6–7.0 GB / 回落到 1.9 GB |

注意：

- 同一时刻只有一份权重在设备上：选 MLX 时 torch 骨干权重被移到主机内存（可换页，不占 wired），
  切回 torch 时再移回设备并释放 MLX 副本；主机内存仍会保留约 1.7 GB。
- 首次启用要花 0.7 秒转换权重，并且每条视频的第一个批次会触发 Metal kernel 预热。
- 张量每次前向要经 CPU 中转（约 15 ms），所以端到端是 1.24 倍左右，而不是骨干单体的 1.3 倍。
- 实现上只做 CPU↔设备搬运，不碰 meta 设备、也不从 checkpoint 重载：早期用 `to_empty("meta")`
  的版本在整模型规模下会让 GPU 卡死（命令提交后等不到完成信号）。
- 多 GPU 的 worker-pool 模式（CUDA）不支持 MLX，会打印警告并继续用 torch。
- 特征缓存不区分后端：已有的 `sam3d_feat*.pt` 会直接复用（两者差异在 0.3% 量级），
  需要强制重算请用 `--overwrite`。

### 内存与 wired 监控

M1/M2 的显存与内存共享，GPU 缓冲属于 **wired（不可换页）内存**。跑大模型时本来就会涨到几 GB，
但如果涨到 20 GB 以上、同时 swap 也开始被吃掉，整机会变得极慢，WindowServer 看门狗可能让系统卡死。

```bash
vm_stat | grep "Pages wired"     # × 16 KB；健康时约 2 GB，跑 SAM 时约 8–10 GB
sysctl vm.swapusage              # used 应为几百 MB
```

注意事项：

- **不要在跑模型时强杀进程**：被 SIGKILL 的进程可能无法归还 GPU 缓冲，wired 会一直占用到重启。
  需要中止时先取消任务、等它自己退出。
- 网页后端正在跑任务时，不要再并行跑命令行或基准脚本，两者各占 4–8 GB。
- wired 一旦被异常占满，只能重启回收；重启后应回落到 2 GB 左右。




## 常见问题

- `RuntimeError: MPS backend out of memory`：降低 SAM 批大小、缩短帧范围，或 `--device cpu`。
- 某些算子提示 MPS 不支持：设置 `PYTORCH_ENABLE_MPS_FALLBACK=1` 后重试（会自动回退到 CPU；
  `app.py` 与命令行脚本已默认开启）。
- `Cannot convert a MPS Tensor to float64 dtype`：说明 MHR 正演被移到 MPS 上执行，
  确认使用本仓库的 `third_party/sam-3d-body`（其中已固定 MHR 在 CPU）。
- 模型校验失败：删除对应的 `.part` 文件后重新运行下载脚本；SMPL-H/MANO 只校验文件结构和大小。
- 端口被占用：`FLOWHMR_PORT=8090 tools/macos/run_web.sh`。
