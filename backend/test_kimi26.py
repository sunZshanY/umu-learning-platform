# -*- coding: utf-8 -*-
"""测试 kimi-k2.6 模型（temperature=1）"""
import asyncio
import os
import sys

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

sys.path.insert(0, os.path.dirname(__file__))
from ai_service import AIService

async def test():
    api_key = os.getenv("UMU_AI_API_KEY")
    base_url = os.getenv("UMU_AI_BASE_URL", "https://api.moonshot.cn/v1")
    model = "kimi-k2.6"

    print("=" * 50)
    print("测试模型: kimi-k2.6 (temperature=1)")
    print("=" * 50)

    service = AIService(api_key=api_key, base_url=base_url, model=model)

    # 手动测试，temperature=1
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=15.0, max_retries=0)

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "回复'OK'，不要其他内容。"}
            ],
            temperature=1.0,
            max_tokens=10,
        )
        text = response.choices[0].message.content or ""
        print(f"[OK] 连接成功！")
        print(f"     模型响应: {text}")
    except Exception as e:
        print(f"[FAIL] 失败: {e}")

    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(test())
