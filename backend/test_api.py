# -*- coding: utf-8 -*-
"""快速测试 Moonshot Kimi API 连接"""
import asyncio
import os
import sys

# 加载 .env
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

sys.path.insert(0, os.path.dirname(__file__))
from ai_service import AIService

async def test():
    api_key = os.getenv("UMU_AI_API_KEY")
    base_url = os.getenv("UMU_AI_BASE_URL", "https://api.moonshot.cn/v1")
    model = os.getenv("UMU_AI_MODEL", "kimi-k2-6")

    print("=" * 50)
    print("UMU 学习平台 - API 连接测试")
    print("=" * 50)
    print(f"服务商: Moonshot (Kimi)")
    print(f"Base URL: {base_url}")
    print(f"模型: {model}")
    print(f"API Key: {api_key[:10]}...{api_key[-6:]}")
    print("=" * 50)

    if not api_key:
        print("[X] 错误: UMU_AI_API_KEY 未设置")
        return

    service = AIService(api_key=api_key, base_url=base_url, model=model)
    print("正在测试连接...")

    connected, msg = await service.test_connection()

    if connected:
        print(f"[OK] 连接成功！")
        print(f"     响应: {msg}")
    else:
        print(f"[FAIL] 连接失败")
        print(f"       原因: {msg}")

    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(test())
