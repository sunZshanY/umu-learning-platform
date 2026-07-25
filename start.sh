#!/bin/bash
# ─────────────────────────────────────────────────────
# UMU刷题助手 - Linux/macOS 启动脚本
# ─────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║         UMU刷题助手 - 启动脚本 v1.0.0                    ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  个人版:  http://localhost:8000                         ║"
echo "║  公开版:  http://localhost:8000/public                  ║"
echo "║  API文档: http://localhost:8000/api/docs                ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# 检查Python
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "[ERROR] Python 未安装，请安装 Python 3.10+"
    exit 1
fi

PYTHON=$(command -v python3 || command -v python)
echo "[INFO] 使用 Python: $($PYTHON --version)"
echo ""

# 安装依赖
echo "[INFO] 安装依赖..."
$PYTHON -m pip install -r "$SCRIPT_DIR/backend/requirements.txt" -q
echo "[INFO] 依赖安装完成"
echo ""

# 检查端口
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "[WARNING] 端口 8000 已被占用，尝试使用端口 8001"
    export PORT=8001
fi

# 启动服务
echo "[INFO] 启动后端服务..."
echo "[INFO] 按 Ctrl+C 停止服务"
echo ""

cd "$SCRIPT_DIR/backend"
$PYTHON server.py
