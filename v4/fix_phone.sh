#!/data/data/com.termux/files/usr/bin/bash
# ============================================================
#  手机训练性能修复（CPU 降频问题）
# ============================================================
set -u

echo "══════════════════════════════════════════════════"
echo "  2048 训练 · CPU 降频修复"
echo "══════════════════════════════════════════════════"
echo ""

# === 核心修复：申请 wake-lock ===
echo "【关键修复】申请 Termux 唤醒锁"
if ! command -v termux-wake-lock >/dev/null 2>&1; then
    echo "  ⚠️ 需要先安装 termux-api"
    echo "     pkg install termux-api"
    echo "     （并在手机上安装 Termux:API App）"
else
    termux-wake-lock
    echo "  ✓ 已获取唤醒锁"
    echo "    → Android 将不再限制 CPU 频率"
fi
echo ""

# === 检查频率变化 ===
echo "【频率检查】"
for i in 7 3 0; do
  MAX=$(cat /sys/devices/system/cpu/cpu$i/cpufreq/cpuinfo_max_freq 2>/dev/null || echo 0)
  CUR=$(cat /sys/devices/system/cpu/cpu$i/cpufreq/scaling_cur_freq 2>/dev/null || echo 0)
  if [ "$MAX" != "0" ]; then
    PCT=$((CUR * 100 / MAX))
    NAME="cpu$i"
    [ $i -eq 7 ] && NAME="大核"
    [ $i -eq 3 ] && NAME="中核"
    [ $i -eq 0 ] && NAME="小核"
    printf "  %-5s %dMHz / %dMHz (%d%%)\n" "$NAME" $((CUR/1000)) $((MAX/1000)) $PCT
  fi
done
echo ""

# === 系统设置提醒 ===
echo "【还需手动设置（一次性）】"
echo "  1. 设置 → 电池 → 省电模式 → 关闭"
echo "  2. 设置 → 应用 → Termux → 电池 → 选「无限制」"
echo "  3. 建议插着充电器跑"
echo ""

# === 启动训练 ===
echo "══════════════════════════════════════════════════"
echo "  启动训练（不绑定 CPU，worker=4）"
echo "══════════════════════════════════════════════════"
echo ""

cd ~/2048 2>/dev/null || cd ~/2048_phone 2>/dev/null || {
    echo "  ✗ 找不到项目目录"
    exit 1
}

# 用新配置：不绑定 + worker 4
nohup python mobile_train.py --workers 4 --no-pin --out-dir ~/2048_train > ~/train.log 2>&1 &
echo "  ✓ 训练已启动（后台）"
sleep 15

echo ""
echo "【15 秒后速度检查】"
python -c "
import json, os
p = os.path.expanduser('~/2048_train/status.json')
if os.path.exists(p):
    d = json.load(open(p))
    r = d.get('episodes_per_sec', 0)
    n = d.get('episode', 0)
    print(f'  局数 {n:,} | 速度 {r:.1f} 局/秒')
    if r > 20:
        print('  ✓ 速度正常')
    elif r > 10:
        print('  △ 仍偏慢，检查是否插电/关省电')
    else:
        print('  ✗ 仍很慢，可能：')
        print('     - termux-api 未安装（无法获取唤醒锁）')
        print('     - 省电模式未关闭')
        print('     - 手机过热降频')
else:
    print('  status.json 未生成，训练可能启动失败')
    print('  查看日志: tail -20 ~/train.log')
"
echo ""
echo "【查看实时日志】"
echo "  tail -f ~/train.log"
echo ""

