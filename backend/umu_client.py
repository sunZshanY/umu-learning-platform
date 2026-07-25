"""
UMU API客户端 — 解析UMU答题会话、获取题目、提交答案
基于对UMU移动端API的逆向分析，提供会话管理功能

注意：UMU的API需要登录认证。本模块提供：
1. 会话解析 — 从URL提取session_id
2. 题目获取 — 通过用户提供的cookie/header代理请求
3. 答案提交 — 代理提交答案到UMU
"""
import re
import json
import logging
import time
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from urllib.parse import urlparse, parse_qs

import httpx

logger = logging.getLogger(__name__)

# ── UMU API端点映射（通过分析移动端行为得出）─────────────────
UMU_BASE = "https://m.umu.cn"
UMU_API_BASE = "https://m.umu.cn"

# 已知的UMU API端点模式
ENDPOINTS = {
    "session_info": "/model/session/quiz/info",
    "question": "/model/session/quiz/question",
    "submit_answer": "/model/session/quiz/submit",
    "session_result": "/model/session/quiz/result",
}

# ── 浏览器UA（模拟微信/移动端）─────────────────────────────
MOBILE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148 "
        "MicroMessenger/8.0.42"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "X-Requested-With": "XMLHttpRequest",
}

REQUEST_TIMEOUT = 12.0
MAX_RETRIES = 2


@dataclass
class UmuSession:
    """UMU答题会话数据类"""
    session_id: str
    url: str
    title: str = ""
    total_questions: int = 0
    current_index: int = 0
    time_limit_seconds: int = 180  # 默认3分钟
    max_wrong_attempts: int = 3
    questions: List[Dict] = field(default_factory=list)
    wrong_count: int = 0
    is_active: bool = True
    created_at: float = field(default_factory=time.time)


class UmuClient:
    """
    UMU客户端

    封装与UMU平台的交互逻辑：
    - URL解析
    - 会话获取
    - 题目抓取
    - 答案提交
    - 错误处理与重试
    """

    def __init__(self, timeout: float = REQUEST_TIMEOUT):
        self.timeout = timeout
        self._sessions: Dict[str, UmuSession] = {}

    # ── URL解析 ──────────────────────────────────────────────
    @staticmethod
    def parse_session_url(url: str) -> Tuple[Optional[str], Optional[str]]:
        """
        从UMU URL中提取session_id和类型

        UMU URL格式：
        - https://m.umu.cn/session/quiz/{session_id}?fwx=1
        - https://m.umu.cn/session/exam/{session_id}
        - https://m.umu.cn/session/survey/{session_id}

        Returns:
            (session_id, session_type) 或 (None, None)
        """
        patterns = [
            r"/session/(quiz|exam|survey)/([A-Za-z0-9_\-]+)",
            r"/session/([A-Za-z0-9_\-]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                groups = match.groups()
                if len(groups) == 2:
                    return groups[1], groups[0]  # session_id, type
                else:
                    return groups[0], "unknown"

        # 尝试从查询参数提取
        try:
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            if "session_id" in params:
                return params["session_id"][0], "quiz"
            if "quiz_id" in params:
                return params["quiz_id"][0], "quiz"
        except Exception:
            pass

        return None, None

    # ── 会话管理 ─────────────────────────────────────────────
    def get_session(self, session_id: str) -> Optional[UmuSession]:
        """获取已缓存的会话"""
        return self._sessions.get(session_id)

    def create_session(self, url: str) -> Tuple[Optional[UmuSession], Optional[str]]:
        """
        创建新的UMU会话

        Args:
            url: UMU答题URL

        Returns:
            (session, error) - 成功时error为None
        """
        session_id, session_type = self.parse_session_url(url)
        if not session_id:
            return None, f"无法从URL解析会话ID: {url}"

        if session_id in self._sessions:
            return self._sessions[session_id], None

        session = UmuSession(
            session_id=session_id,
            url=url,
        )
        self._sessions[session_id] = session
        logger.info(f"创建UMU会话: {session_id} (类型: {session_type})")
        return session, None

    # ── API请求封装 ──────────────────────────────────────────
    async def _request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        cookies: Optional[Dict] = None,
        method: str = "GET",
    ) -> Tuple[Optional[Dict], Optional[str]]:
        """
        发送UMU API请求（带重试）

        Returns:
            (response_data, error)
        """
        url = f"{UMU_API_BASE}{endpoint}"
        req_headers = {**MOBILE_HEADERS}
        if headers:
            req_headers.update(headers)

        last_error = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    if method == "GET":
                        resp = await client.get(
                            url, params=params, headers=req_headers, cookies=cookies
                        )
                    else:
                        req_headers["Content-Type"] = "application/json"
                        resp = await client.post(
                            url, json=data, headers=req_headers, cookies=cookies
                        )

                    if resp.status_code == 200:
                        try:
                            result = resp.json()
                            if result.get("status") == 0 or result.get("errno") == 0:
                                return result, None
                            else:
                                err_msg = result.get("error") or result.get("error_msg") or "未知错误"
                                return None, f"API错误 [errno={result.get('errno')}]: {err_msg}"
                        except json.JSONDecodeError:
                            return None, f"响应非JSON格式: {resp.text[:200]}"
                    elif resp.status_code == 401 or resp.status_code == 403:
                        return None, "需要登录认证，请提供有效的session cookie"
                    elif resp.status_code == 404:
                        return None, f"接口不存在 (404): {endpoint}"
                    elif resp.status_code >= 500:
                        last_error = f"服务器错误 ({resp.status_code})"
                        if attempt < MAX_RETRIES:
                            await self._sleep(1.0 * (attempt + 1))
                        continue
                    else:
                        return None, f"HTTP {resp.status_code}: {resp.text[:200]}"

            except httpx.TimeoutException:
                last_error = f"请求超时 ({self.timeout}s)"
                if attempt < MAX_RETRIES:
                    await self._sleep(1.0 * (attempt + 1))
            except httpx.ConnectError as e:
                last_error = f"连接失败: {str(e)[:100]}"
                if attempt < MAX_RETRIES:
                    await self._sleep(1.5 * (attempt + 1))
            except Exception as e:
                last_error = f"请求异常: {str(e)[:150]}"
                if attempt < MAX_RETRIES:
                    await self._sleep(1.0 * (attempt + 1))

        return None, last_error or "未知请求错误"

    async def _sleep(self, seconds: float):
        """异步sleep的helper"""
        import asyncio
        await asyncio.sleep(seconds)

    # ── 题目获取 ─────────────────────────────────────────────
    async def fetch_questions(
        self,
        session_id: str,
        cookies: Optional[Dict] = None,
        headers: Optional[Dict] = None,
    ) -> Tuple[Optional[List[Dict]], Optional[str]]:
        """
        从UMU获取题目列表

        Args:
            session_id: 会话ID
            cookies: 用户登录后的cookie字典
            headers: 额外的请求头

        Returns:
            (questions_list, error)
        """
        # 尝试多个可能的API端点
        endpoints_to_try = [
            f"{ENDPOINTS['session_info']}?session_id={session_id}",
            f"/api/v1/session/{session_id}/questions",
            f"/model/session/{session_id}/questions",
        ]

        for endpoint in endpoints_to_try:
            result, error = await self._request(
                endpoint,
                params={"session_id": session_id},
                cookies=cookies,
                headers=headers,
            )
            if result:
                data = result.get("data", result)
                questions = self._extract_questions(data)
                if questions:
                    return questions, None

        return None, "无法获取题目：可能需要登录认证，请提供UMU登录后的cookie"

    @staticmethod
    def _extract_questions(data: Any) -> List[Dict]:
        """
        从UMU响应中提取题目列表

        UMU的响应结构各有不同，需要兼容多种格式
        """
        questions = []

        if isinstance(data, list):
            # 直接是题目列表
            for item in data:
                if isinstance(item, dict) and "question" in item:
                    questions.append(item)
        elif isinstance(data, dict):
            # 嵌套结构
            for key in ("questions", "quiz_list", "items", "list", "data"):
                if key in data and isinstance(data[key], list):
                    for item in data[key]:
                        if isinstance(item, dict) and ("question" in item or "title" in item or "content" in item):
                            questions.append(item)
                    if questions:
                        break

        return questions

    # ── 答案提交 ─────────────────────────────────────────────
    async def submit_answer(
        self,
        session_id: str,
        question_id: str,
        answer: Any,
        cookies: Optional[Dict] = None,
        headers: Optional[Dict] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        提交答案到UMU

        Returns:
            (success, error)
        """
        data = {
            "session_id": session_id,
            "question_id": question_id,
            "answer": answer,
        }

        result, error = await self._request(
            ENDPOINTS["submit_answer"],
            data=data,
            method="POST",
            cookies=cookies,
            headers=headers,
        )

        if result:
            is_correct = result.get("data", {}).get("is_correct", False) or result.get("is_correct", False)
            return is_correct, None

        return False, error

    # ── 会话状态追踪 ─────────────────────────────────────────
    def record_answer_result(self, session_id: str, is_correct: bool):
        """记录答题结果，追踪错误次数"""
        session = self.get_session(session_id)
        if session:
            session.current_index += 1
            if not is_correct:
                session.wrong_count += 1
                if session.wrong_count >= session.max_wrong_attempts:
                    session.is_active = False
                    logger.warning(
                        f"会话 {session_id}: 错误次数达到上限"
                        f"({session.wrong_count}/{session.max_wrong_attempts})"
                    )

    def get_remaining_attempts(self, session_id: str) -> int:
        """获取剩余答题机会"""
        session = self.get_session(session_id)
        if not session:
            return 0
        return max(0, session.max_wrong_attempts - session.wrong_count)

    def should_stop(self, session_id: str) -> bool:
        """检查是否应该停止答题（错误次数已达上限）"""
        session = self.get_session(session_id)
        if not session:
            return True
        return not session.is_active or self.get_remaining_attempts(session_id) <= 0
