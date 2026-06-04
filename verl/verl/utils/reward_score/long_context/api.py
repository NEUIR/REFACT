from openai import OpenAI
import os
import inspect
from .logger import cost_logger

_sync_client = None

# ============================================================
#  超时与默认配置
# ============================================================
DEFAULT_API_TIMEOUT = 90          # 单次请求超时 (秒)
DEFAULT_MAX_RETRIES = 3           # 最大重试次数
DEFAULT_RETRY_BASE_SLEEP = 5     # 重试基础等待时间 (秒)


def _get_caller_function_name():
    """Get the name of the function that called call_api"""
    for frame_info in inspect.stack()[2:]:
        func_name = frame_info.function
        if func_name not in ('wrapper', '<module>', '_call_api_internal'):
            return func_name
    return "unknown"


def load_api_key():
    return os.getenv("YEY_API_KEY")


def get_sync_client():
    global _sync_client
    if _sync_client is None:
        _sync_client = OpenAI(
            api_key="sk-sEKt8LeFPmbgT6kVEcE3B80bEe064bBd9e3e8bF6F4026e6c",
            base_url="https://api.shubiaobiao.cn/v1",
            timeout=DEFAULT_API_TIMEOUT,       # ★ 连接+读取总超时
            max_retries=0,                      # ★ 关闭 SDK 内置重试，由我们自己控制
        )
    return _sync_client


def call_api(model, contents, caller_name=None, temperature=1.0,
             system_prompt=None, timeout=None):
    """
    Single request version of the API call.

    Args:
        model: 模型名称
        contents: str (单条 user 消息) 或 list[dict] (完整 messages 列表)
        caller_name: 调用方函数名 (用于日志)
        temperature: 温度参数
        system_prompt: 可选的 system message
        timeout: 可选，覆盖默认超时 (秒)。传 None 使用 DEFAULT_API_TIMEOUT

    Returns:
        模型回复文本；所有重试均失败时返回空字符串而非抛异常，
        避免在 RL reward 计算中导致整个训练 hang 或崩溃。
    """
    client = get_sync_client()

    # ---------- 构造 messages ----------
    if isinstance(contents, str):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": contents})
    elif isinstance(contents, list):
        if system_prompt:
            has_system = any(m.get("role") == "system" for m in contents)
            if has_system:
                messages = contents
            else:
                messages = [{"role": "system", "content": system_prompt}] + contents
        else:
            messages = contents
    else:
        raise ValueError(f"contents 类型不支持: {type(contents)}")

    # ---------- 请求级超时 ----------
    request_timeout = timeout or DEFAULT_API_TIMEOUT

    # ---------- Retry loop ----------
    last_exception = None

    for attempt in range(DEFAULT_MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                timeout=request_timeout,        # ★ 每次请求的超时
            )

            # Extract token usage
            input_tokens = 0
            output_tokens = 0
            if hasattr(completion, 'usage') and completion.usage:
                input_tokens = getattr(completion.usage, 'prompt_tokens', 0) or 0
                output_tokens = getattr(completion.usage, 'completion_tokens', 0) or 0
            else:
                print(f"⚠️ [API] No usage info found in response from {model}")

            function_name = caller_name or _get_caller_function_name()
            cost_logger.log(model, function_name, input_tokens, output_tokens)

            if completion.choices[0].message.content is None:
                print(f"No response from {model}")
                return ""                       # ★ 返回空串，不返回 "No response"
            return completion.choices[0].message.content

        except Exception as e:
            last_exception = e
            import time
            error_type = type(e).__name__
            print(f"⚠️ [API] {error_type} on attempt {attempt + 1}/{DEFAULT_MAX_RETRIES}: {e}")
            if attempt < DEFAULT_MAX_RETRIES - 1:
                sleep_time = DEFAULT_RETRY_BASE_SLEEP * (2 ** attempt)
                print(f"Waiting {sleep_time}s before retry...")
                time.sleep(sleep_time)

    # ★ 所有重试失败：返回空串而非抛异常，保证训练不会 hang
    print(f"❌ [API] All {DEFAULT_MAX_RETRIES} retries failed. "
          f"Last error: {type(last_exception).__name__}: {last_exception}")
    return ""