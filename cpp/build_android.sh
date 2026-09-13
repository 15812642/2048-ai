#!/bin/bash
# TDL2048+ Android 构建（Termux / bionic libc）
# 用法: bash build_android.sh [源码目录]
#
# 前置条件:
#   - Android NDK（r26+，本文档验证 r29-beta1）
#   - 已应用两个补丁: patches/shm-android.patch + patches/clang20-constexpr.patch
#
set -e
SRC="${1:-.}"
NDK="${ANDROID_NDK_HOME:-/path/to/android-ndk}"
CXX="$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android21-clang++"

cd "$SRC"

# 1. 应用补丁（如果尚未应用）
if ! grep -q "SHM_MMAP" moporgic/shm.h 2>/dev/null; then
    echo "→ 应用 shm-android.patch ..."
    patch -p1 < patches/shm-android.patch || {
        echo "⚠️ patch 命令失败，可手动按照 patch 内容修改 moporgic/shm.h"
        exit 1
    }
fi
if grep -q "constexpr inline expectimax" 2048.cpp 2>/dev/null; then
    echo "→ 应用 clang20-constexpr.patch ..."
    sed -i 's/constexpr inline expectimax(utils::options::option opt)/inline expectimax(utils::options::option opt)/' 2048.cpp
fi

# 2. 编译（-DSHM_MMAP 启用 mmap 共享内存分支）
echo "→ 交叉编译（bionic）..."
"$CXX" -O3 -std=c++17 -pthread -DSHM_MMAP -static-libstdc++ \
       -Wno-psabi -fmessage-length=0 \
       -o 2048-arm64 2048.cpp

echo "✓ 编译完成: $PWD/2048-arm64"
echo ""
echo "验证（应无 clone3 符号、依赖仅 libc/libm/libdl）:"
echo "  nm 2048-arm64 | grep clone3   # 应无输出"
echo "  readelf -d 2048-arm64 | grep NEEDED"
