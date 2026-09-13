#!/data/data/com.termux/files/usr/bin/bash
# ============================================================
#  手机训练性能优化脚本（解决 CPU 降频问题）
# ============================================================
set -u

echo "════════════════════════════════════════════════════════"
echo "  2048 训练 · 手机性能优化"
echo "════════════════════════════════════════════════════════"
echo ""

# ---- 1. CPU 频率检查 ----
echo "【1】当前 CPU 频率"
for i in 0 3 7; do
  MAX=$(cat /sys/devices/system/cpu/cpu$i/cpufreq/cpuinfo_max_freq 2>/dev/null || echo 0)
  CUR=$(cat /sys/devices/system/cpu/cpu$i/cpufreq/scaling_cur_freq 2>/dev/null || echo 0)
  if [ "$MAX" != "0" ]; then
    PCT=$((CUR * 100 / MAX))
    TAG=""
    [ $i -eq 7 ] && TAG="大核"
    [ $i -eq 3 ] && TAG="中核"
    [ $i -eq 0 ] && TAG="小核"
    printf "   cpu%d (%s): %dMHz / %dMHz (%d%%)\n" $i "$TAG" $((CUR/1000)) $((MAX/1000)) $PCT
  fi
done
echo ""

# ---- 2. 申请唤醒锁（关键！防止系统休眠 CPU）----
echo "【2】申请 Termux 唤醒锁（防止 CPU 被限制）"
if command -v termux-wake-lock >/dev/null 2>&1; then
  termux-wake-lock
  echo "   ✓ 已获取 Wake Lock（保持 CPU 活跃）"
else
  echo "   ⚠️ 未安装 termux-api"
  echo "      请先执行: pkg install termux-api"
  echo "      并在手机上安装 Termux:API App"
fi
echo ""

# ---- 3. 提示系统设置 ----
echo "【3】需要手动做的系统设置（重要！）"
echo "   ▸ 关闭省电模式: 设置 → 电池 → 省电模式 → 关闭"
echo "   ▸ 关闭智能省电: 设置 → 电池 → 后台耗电管理 → Termux → 无限制"
echo "   ▸ 开发者选项 → 关闭「强制 GPU 渲染」（若有）"
echo "   ▸ 最好是【插着充电器】跑训练"
echo "   ▸ 手机放在通风处（别放床上/被子里）"
echo ""

# ---- 4. 检查温度（若可读）----
echo "【4】温度检查"
for zone in /sys/class/thermal/thermal_zone*/temp; do
  T=$(cat $zone 2>/dev/null)
  if [ -n "$T" ] && [ "$T" -gt 0 ] 2>/dev/null; then
    C=$((T / 1000))
    if [ $C -gt 30 ] && [ $C -lt 120 ]; then
      WARN=""
      [ $C -gt 60 ] && WARN="  ⚠️ 过热，会降频！"
      echo "   $(basename $(dirname $zone)): ${C}°C$WARN"
      break
    fi
  fi
done
echo ""

# ---- 5. 提供优化后的启动命令 ----
echo "【5】推荐启动配置"
echo ""
echo "   方案 A（推荐，减少发热）："
echo "     python mobile_train.py --workers 3 --no-pin --out-dir ~/2048_train"
echo ""
echo "   方案 B（保留绑定，但降低 worker）："
echo "     python mobile_train.py --workers 4 --out-dir ~/2048_train"
echo ""
echo "   方案 C（性能优先，插电+通风）："
echo "     python mobile_train.py --workers 5 --no-pin --out-dir ~/2048_train"
echo ""
echo "   ★ --no-pin 表示不绑定 CPU，让系统调度器自由分配"
echo "     （绑定反而可能因为核心少而更慢）"
echo ""
echo "════════════════════════════════════════════════════════"
