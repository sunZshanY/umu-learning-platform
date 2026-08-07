# -*- coding: utf-8 -*-
"""测试 Moonshot API 不同模型名称"""
import asyncio
import os
import sys

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

sys.path.insert(0, os.path.dirname(__file__))
from ai_service import AIService

async def test_model(model_name):
    api_key = os.getenv("UMU_AI_API_KEY")
    base_url = os.getenv("UMU_AI_BASE_URL", "https://api.moonshot.cn/v1")

    service = AIService(api_key=api_key, base_url=base_url, model=model_name)
    connected, msg = await service.test_connection()
    return connected, msg

async def main():
    candidates = [
        "moonshot-v1-8k",
        "moonshot-v1-32k",
        "moonshot-v1-128k",
        "kimi-k2-6",
        "kimi-k2.6",
        "kimi-k2-6-20250801",
    ]

    print("=" * 60)
    print("测试多个模型名称...")
    print("=" * 60)

    for model in candidates:
        print(f"\n测试模型: {model}")
        try:
            connected, msg = await test_model(model)
            if connected:
                print(f"  [OK] 成功: {msg}")
            else:
                print(f"  [FAIL] 失败: {msg}")
        except Exception as e:
            print(f"  [ERROR] 异常: {e}")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
