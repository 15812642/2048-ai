# 已应用的补丁说明

本目录是 [moporgic/TDL2048+](https://github.com/moporgic/TDL2048) 的**快照副本**，
已应用以下两个补丁（上游版本分别为 Rev.2021 与 clang-20 不兼容问题）。

## 补丁 1：Android 共享内存移植（shm.h）

**文件**：`moporgic/shm.h`（+57 行）

**背景**：Android 内核不支持 System V 共享内存（`shmget` 返回
`Function not implemented`），多线程训练模式无法启动。

**方案**：新增 `SHM_MMAP` 编译分支，用 `mmap(MAP_SHARED | MAP_ANONYMOUS)`
替代 SysV shm。可行性的关键：框架的 fork 时序（`alloc` 先于 `fork`，
子进程继承映射）与 MAP_SHARED 匿名映射语义完全匹配。

启用方式：编译时加 `-DSHM_MMAP`。

## 补丁 2：clang-20 严格性修复（2048.cpp）

**文件**：`2048.cpp` 第 1617 行（1 行修改）

```diff
-		constexpr inline expectimax(utils::options::option opt) {
+		inline expectimax(utils::options::option opt) {
```

**背景**：clang-20 拒绝 `constexpr` 构造函数带非字面类型参数
（`utils::options::option` 非 literal type），gcc 较宽松故上游未暴露。

## 如何自行复现这两个补丁

见上一级目录的 `patches/`（标准 unified diff，可直接 `patch -p1`）。

## 编译验证

```bash
# 服务器/桌面（x86_64）
g++ -O3 -std=c++17 -o 2048 2048.cpp

# Android（NDK bionic，必须加 -DSHM_MMAP）
$NDK/toolchains/llvm/prebuilt/*/bin/aarch64-linux-android21-clang++ \
  -O3 -std=c++17 -pthread -DSHM_MMAP -static-libstdc++ \
  -o 2048-arm64 2048.cpp
```

## 许可

上游为 MIT 许可（见 `LICENSE.md`），本副本保留原始版权声明。
