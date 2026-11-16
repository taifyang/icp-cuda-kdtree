# ICP GPU 优化记录

环境：RTX 4090 / CUDA 12.8 / driver 580，数据为 `data/` 下 4 个点云（bunny 35947 点、table_scene 460400 点）。
下列耗时均为**稳态工作耗时**，即已排除一次性 CUDA 上下文初始化（见 §4.2）。

## 1. 先把程序跑起来（最容易踩的空坑）

- **`data/` 下的 `.pcd` 全部是 `DATA binary`，而 `main.cpp` 用 `>>` 按 ASCII 读取**，且固定读 6 列（x y z r g b），bunny 只有 3 列。直接喂自带数据得不到结果，任何性能数字都无从谈起。目前没有二进制解析器。
- 依赖 `libeigen3-dev`，`CMakeLists.txt` 里路径硬编码为 `/usr/include/eigen3`。
- 构建：`cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89 && cmake --build build -j8`

## 2. 结果

| 数据集 | 优化前 | 优化后 | 提速 | 残差（越小越好） |
|---|---|---|---|---|
| bunny 36k | 14.5 ms / 22 轮 | 6.8 ms / 23 轮 | 2.1x | 6.72e-7 → **6.65e-7** |
| table 460k | 255 ms / 100 轮 | **66.7 ms / 38 轮** | 3.8x | 8.42e-6 → **1.81e-6** |

table 的残差改善不是顺带效果，而是收敛判据修正的直接结果（§4.4）：**优化前返回的是漂移后的差结果**。

单项措施的贡献：

| 措施 | 实测影响 |
|---|---|
| `d_fcp` blockDim 1 → 32 | table 稳态 255 → 150 ms（1.7x）。该 kernel 占 GPU 总时间 71% |
| 收敛判据修正 | 迭代数 100 → 38，且返回最优而非末次 |
| 缓冲区复用 | `cudaMalloc`/`cudaFree` 各 1929 → 27 次 |
| 计时口径修正 | 报告值 395 → 67 ms（消除 150-183 ms 的初始化计入） |

## 3. 稳态时间都花在哪（table，优化前）

`nsys` profile 结论：

- GPU 全部 kernel 合计仅约 81 ms，而 ICP 计时 450 ms —— **九成以上时间不在 GPU 上**。
- `d_fcp` 56.9 ms：blockDim=1，每 warp 只有 1 个活跃线程，内存带宽利用率 1/32。
- `cudaMalloc` + `cudaFree` 各 1929 次、合计约 69 ms（稳态），全部来自 `GPUMatrix` 每个运算新建临时矩阵；`sum_colwise()` 的递归一次产生 3 层分配。
- `zero_kernel` 被调 903 次——`sum_colwise()` 在 `GPUMatrix::zero()` 里多跑了一遍清零，而 `block_sum_kernel` 是**赋值而非累加**，根本不需要清零。
- `divide_kernel` 4.75 ms：`mean()` 先对 N×3 整体除以 N 再做归约，纯属多余。
- `transpose_kernel<<<1,1>>>`、`to_transformation_kernel`、4×4 的 `dot_kernel` 各约 13 µs，全是启动+分配开销，实际只处理 3×3 / 4×4。

## 4. 关键陷阱（长期有效，改代码前请先读）

### 4.1 `GPUMatrix` 是 row-major，Eigen 是 column-major —— 两个隐式转置互相抵消

`GPUMatrix::to_cpu()` 直接把 row-major 的 device 数据 `memcpy` 进 column-major 的 `Eigen::MatrixXf`，读出来就是转置的；`from_cpu()` 同理反向再转一次。因此旧代码里 `transformation.to_cpu().transpose()` 返回的**才是正确的变换矩阵**——那个 `.transpose()` 看着多余，删掉就全错。

重写时已改为：所有小矩阵运算只在 host 上用 Eigen 显式做完，把 12 个 float 当**值参数**传给 kernel，不再经过任何 row/column-major 边界。要恢复这类转换时务必重新验证 `‖T·p − 返回点云‖`。

### 4.2 一次性 CUDA 初始化会污染基准测试

第一笔 `cudaMalloc` 会触发上下文+模块加载，实测 150~183 ms。bunny 真实工作只要 7 ms，旧代码却报告 160 ms（其中 95% 是启动成本）。计时区之前调用 `icp_warmup()`。

另外 `std::chrono::duration<double>::count()` 的单位是**秒**，旧代码直接印成 `"ms"`，虚高 1000 倍。

### 4.3 device 侧浮点 `atomicAdd` 破坏结果可复现性

`fit_kernel` 最初用 shared `atomicAdd` 汇总各 warp 的部分和，导致同一输入多次运行结果不同（旧实现是逐位确定的）。浮点加法不满足结合律，求和顺序一旦依赖调度就不可复现，末位差异还会被迭代放大。

结论：**任何需要可复现的归约都写成 per-warp 部分和 + 固定顺序的串行合并**，不要用浮点 atomic。

### 4.4 固定阈值 + 返回最后一次迭代 = 又慢又不准

table 的误差曲线在**第 40 轮到达最低 1.81e-6，随后单调回升到 100 轮的 8.42e-6**。原因：点对应 ICP 在稠密自相似点云上，对应关系退化后源点云会互相"吸附"漂移。而 `threshold=1e-6` 永远达不到，于是 100 轮全部跑满，**返回的还是最差的那一次**。

现在的停止条件：相对改进 < `relative_threshold` 或误差回升即停，并返回**最优**迭代对应的变换。

推论：`threshold` 不能瞎设——它必须落在数据实际能达到的精度量级上，否则等于关闭了早停。

### 4.5 `cut_off_radius` 设小会锁死在错误局部最优

剔除远距离对应能抗外点，但阈值是**绝对量**：若初始错配比它大得多，前几轮几乎所有对应都被剔除，拟合只依据少量偶然匹配，结果锁死在局部最优。实测 table（初始残差 0.064）：

- cutoff=0.05 → 1.6e-6，正常
- cutoff=0.01 → 停在 0.005 的错误解（5 轮后自认为收敛）
- cutoff=1e-7 → 全部剔除，安全退化为恒等变换（不崩、不泄漏）

固定阈值本质上做不到既安全又通用；真要用得改成自适应阈值或按分位数截断（trimmed ICP）。**默认值保持 `inf`（与原行为一致）是当前正确选择。**

### 4.6 Kabsch/Scott 步必然把源质心映射到目标质心

`t = μ_m − R·μ_p` 意味着 `mean(R·p + t) = μ_m` 恒成立。所以 ICP 循环里每轮重新归约源点云质心是**完全多余的**——第一次之后质心就是目标质心。这个恒等式直接消掉了热循环里的一整轮 N×3 归约。

### 4.7 `cukd` 的 `computeBounds_atomicGrow` 有原子争用（未修）

建树时对**同 6 个地址**做每点 6 次 float CAS（`min/max` 的语义保证结果确定，所以不影响正确性，只是慢）。改成 block 内 shared 归约 + 每 block 一次 atomic 后，460k 点的 table 上建树从 1.49 ms 降为可忽略。这是第三方 vendored 代码，本次未改动。

### 4.8 旧代码的 `d_fcp` 只覆盖 `q.size()` 的缓冲区

`d_results` 按参考点云 `q` 的大小分配，却按 `p` 的规模写入。两份测试数据刚好等长所以没暴露，源点云更大时就是越界写。已改为按 `p.size()` 分配。

## 5. 怎么复现这套验证

```bash
# 性能与 API 调用次数
nsys profile -o /tmp/p --trace=cuda ./build/icp src.xyz tgt.xyz
nsys stats --report cuda_gpu_kern_sum --report cuda_api_sum /tmp/p.nsys-rep

# 泄漏与访存错误（基线：1,294,116 字节 / 4 处分配 / 4 个错误；当前：0 / 0）
compute-sanitizer --tool memcheck --leak-check full ./build/icp src.xyz tgt.xyz

# 结果可复现性：同一输入连跑 4 次，误差序列必须逐位一致
```

独立正确性校验不要只看程序内部打印的 `error`（它是内部定义的量），应该：

1. 穷举最近邻算**返回点云**对目标点云的平均距离（优化后 table = 1.81e-6，未配准 = 0.0643）；
2. 校验返回的矩阵与返回的点云自洽：`max‖T·pᵢ − returnedᵢ‖` 应在 float 舍入量级（实测 2.7e-7）；
3. 检查 `det(R) = 1`，排除镜像反射。

## 6. 尚未处理

- 二进制 PCD 解析（§1，实际阻塞使用）。
- `find_rotation` 缺 `det < 0` 的反射保护；两个数据集实测 det=1，未触发。
- 点对平面（point-to-plane）距离 + 多分辨率 coarse-to-fine：这类扫描数据上收敛显著更快，是下一个数量级的来源。
- `GPUMatrix` 现存一批无调用者的成员（`dot` / `subtract` / `Identity` / `closest` / `to_float3` / `from_cpu` / `to_cpu` / `sum_colwise` / `mean`）——ICP 路径已不经过它们，等确认后清理。
- `CMakeLists.txt` 用 `aux_source_directory` 收集源文件：目录里任何多余的同类型文件都会被编进目标（会在链接期造成重复定义），新增文件时注意。
