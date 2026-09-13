#!/bin/bash
# TDL2048+ 服务器版构建（x86_64 Linux）
# 用法: bash build_server.sh [源码目录]
set -e
SRC="${1:-.}"
cd "$SRC"

# 1. 克隆（如果没有源码）
if [ ! -f 2048.cpp ]; then
    git clone --depth 1 https://github.com/moporgic/TDL2048.git
    cd TDL2048
fi

# 2. 编译（官方默认参数即可；注意 -march=native 反而更慢）
g++ -O3 -std=c++17 -o 2048 2048.cpp

echo "✓ 编译完成: $PWD/2048"
echo ""
echo "启动训练（OTD+TC 配方, 4线程, 每500K局存盘）:"
cat << 'CMD'
./2048 \
  -n "012456=5000 456789=5000 012345=5000 234569=5000 01259a=5000 345678=5000 134567=5000 01489a=5000" \
  -a 0.1 -p 4 -t "LOOP[999999999]" -u 500 -o weights.w
CMD
