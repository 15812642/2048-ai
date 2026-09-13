# C++ 训练器部署（TDL2048+）

本目录收录本项目把官方 C++ 框架 [moporgic/TDL2048+](https://github.com/moporgic/TDL2048)
部署到「x86 服务器 + Android 手机（Termux）」的全部补丁与脚本。

## 为什么用 C++

| 指标 | Python (本仓库 v3) | C++ (TDL2048+) |
|---|---|---|
| 训练速度 | 8.7 局/秒 | ~310 局/秒（4 核中部）|
| 跑 1 亿局（论文标定训练量）| 133 天 | ~1-4 天 |

## 快速开始（服务器）

```bash
git clone --depth 1 https://github.com/moporgic/TDL2048.git
cd TDL2048
g++ -O3 -std=c++17 -o 2048 2048.cpp

# OTD+TC 训练（论文 Table VI 配方）
./2048 \
  -n "012456=5000 456789=5000 012345=5000 234569=5000 01259a=5000 345678=5000 134567=5000 01489a=5000" \
  -a 0.1 -p 4 -t "LOOP[999999999]" -u 500 -o weights.w
```

> ⚠️ **关键坑**：`-t LOOP[N]` 语法无效（help 文档误导）。
> 实际每循环目标 = `1000 × unit / 线程数`（每线程）。用 `-u 500` → 每循环全局 500K 局存盘一次。

## Android（Termux）移植

三个坑的完整解决（详见 `../docs/DEVELOPMENT_LOG.md` 第七章）：

1. **glibc 静态二进制 → SIGSYS**：glibc 的 `clone3` 被 Android seccomp KILL
   → 必须用 NDK bionic libc 交叉编译
2. **Android 无 SysV 共享内存**：`shmget` = `Function not implemented`
   → `patches/shm-android.patch` 提供 `mmap(MAP_SHARED|MAP_ANONYMOUS)` 分支
3. **clang-20 严格性**：`constexpr` 构造非字面类型参数报错
   → `patches/clang20-constexpr.patch`（一行）

```bash
# 打补丁 + 交叉编译
patch -p1 < patches/shm-android.patch
sed -i 's/constexpr inline expectimax/inline expectimax/' 2048.cpp
$NDK/toolchains/llvm/prebuilt/*/bin/aarch64-linux-android21-clang++ \
  -O3 -std=c++17 -pthread -DSHM_MMAP -static-libstdc++ -o 2048-arm64 2048.cpp
```

**验证清单**：
```bash
nm 2048-arm64 | grep clone3        # 必须无输出
readelf -d 2048-arm64 | grep NEEDED # 只有 libc/libm/libdl
readelf -p .interp 2048-arm64      # /system/bin/linker64
```

## 三节点 Ensemble（权重合并）

```bash
# 服务器 + 手机A + 手机B 各跑一份后合并
./2048 -n 8x6patt -e 3 -p 4 -i srv.w phoneA.w phoneB.w -o merged.w
```

官方机制：同签名的权重表自动取平均。

## 实测性能（本项目）

| 设备 | 配置 | 吞吐 |
|---|---|---|
| 云服务器 4 核 | 4 线程 | ~310 局/秒（中期）|
| vivo MT6785（2019 中端）| 8 线程全核 | 561~1150 局/秒 |
| 骁龙 8 Gen 2 | 5 线程 | ~900 局/秒 |
