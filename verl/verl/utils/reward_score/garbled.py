import re
from concurrent.futures import ProcessPoolExecutor

garbled_executor = ProcessPoolExecutor(max_workers=100)


def batch_is_garbled(tokenizer, responses, threshold: float = 0.1) -> bool:
    ended_tags = [tokenizer.eos_token_id in resp for resp in responses]

    # Submit garbled checks to process pool
    response_strings = tokenizer.batch_decode(
        responses, skip_special_tokens=True
    )
    garbled_futures = [
        garbled_executor.submit(_is_garbled, resp, threshold)
        for resp in response_strings
    ]

    # Get results from futures
    garbled_tags = [future.result() for future in garbled_futures]

    return [
        float((not ended) or garbled)
        for ended, garbled in zip(ended_tags, garbled_tags)
    ]


def is_garbled(tokenizer, response, threshold: float = 0.1) -> bool:
    """
    判断字符串是否
    1. 没有正常结束
    2. 包含大量乱码（中文不算乱码）。

    Args:
        threshold: 乱码字符比例阈值（默认0.3，即30%）

    Returns:
        bool: 如果乱码比例超过阈值，返回True，否则返回False
    """
    if tokenizer.eos_token_id not in response:
        # print(f"is_garbled, no eos token: {response}")
        return True
    text = tokenizer.decode(response, skip_special_tokens=True)
    # print(f"is_garbled, text: {text}")
    if not text:
        return True

    # 定义中文字符范围（包括常用汉字）
    chinese_pattern = re.compile(r"[\u4e00-\u9fff]")
    # 定义常见ASCII字符（字母、数字、常见符号等）
    ascii_pattern = re.compile(
        r'[a-zA-Z0-9\s.,!?@#$%^&*()_+\-=\[\]{}\\|;:"\'<>/?`~]'
    )

    total_length = len(text)
    # 计算中文字符数量
    chinese_count = len(chinese_pattern.findall(text))
    # 计算常见ASCII字符数量
    ascii_count = len(ascii_pattern.findall(text))

    # 乱码字符数 = 总长度 - 中文字符数 - 常见ASCII字符数
    garbled_count = total_length - chinese_count - ascii_count

    # 计算乱码比例
    garbled_ratio = garbled_count / total_length if total_length > 0 else 0

    return garbled_ratio > threshold


def _is_garbled(text, threshold: float = 0.3) -> bool:
    """
    判断字符串是否
    1. 没有正常结束
    2. 包含大量乱码（中文不算乱码）。
    3. 中英混杂。

    Args:
        threshold: 乱码字符比例阈值（默认0.3，即30%）

    Returns:
        bool: 如果乱码比例超过阈值，返回True，否则返回False
    """
    if not text:
        return False

    # 定义中文字符范围（包括常用汉字）
    chinese_pattern = re.compile(r"[\u4e00-\u9fff]")
    # 定义常见ASCII字符（字母、数字、常见符号等）
    ascii_pattern = re.compile(
        r'[a-zA-Z0-9\s.,!?@#$%^&*()_+\-=\[\]{}\\|;:"\'<>/?`~]'
    )

    total_length = len(text)
    # 计算中文字符数量
    chinese_count = len(chinese_pattern.findall(text))
    # 计算常见ASCII字符数量
    ascii_count = len(ascii_pattern.findall(text))

    # 乱码字符数 = 总长度 - 中文字符数 - 常见ASCII字符数
    garbled_count = total_length - chinese_count - ascii_count

    # 计算乱码比例
    garbled_ratio = garbled_count / total_length if total_length > 0 else 0

    return garbled_ratio > threshold


def is_language_mix(text, mix_threshold=0.4):
    """
    检测字符串中中文和英文的比例，如果中英混输比例过高则报错。

    参数:
        text (str): 输入字符串
        mix_threshold (float): 中英混输比例阈值（0到1之间），默认0.3

    返回:
        dict: 包含中文、英文、其他字符的比例

    抛出:
        ValueError: 如果中英混输比例过高
    """
    if not text:
        return {"chinese": 0.0, "english": 0.0, "other": 0.0}

    # 使用正则表达式匹配中文和英文
    chinese_pattern = re.compile(r"[\u4e00-\u9fff]")  # 匹配中文字符
    english_pattern = re.compile(r"[a-zA-Z]")  # 匹配英文字符

    # 计算各种字符的个数
    total_length = len(text)
    chinese_count = len(chinese_pattern.findall(text))
    english_count = len(english_pattern.findall(text))
    other_count = total_length - chinese_count - english_count

    # 计算比例
    chinese_ratio = chinese_count / total_length if total_length > 0 else 0
    english_ratio = english_count / total_length if total_length > 0 else 0
    other_ratio = other_count / total_length if total_length > 0 else 0

    # 检查中英混输比例
    # 如果中文和英文都有，且两者比例都不为0，检查是否超过阈值
    if chinese_ratio > 0 or english_ratio > 0 or other_ratio > 0:
        mix_ratio = min(chinese_ratio, english_ratio) / (
            chinese_ratio + english_ratio
        )
        if mix_ratio > mix_threshold:
            return True
    return False


# 测试用例
if __name__ == "__main__":
    test_cases = [
        "Hello, World!",  # 正常
        "aaaaa bbbbb ccccc",  # 重复字符
        "¡¢£¤¥¦§¨©ª«¬®¯°±²³",  # 非ASCII乱码
        "x9k#2m$pQz@7vL&",  # 高熵随机字符串
        "ababababababab",  # 重复模式
        "!!!!!!!",  # 全是标点
        "",  # 空字符串
        "正常中文文本，带点English",  # 正常中英混合
    ]

    for test in test_cases:
        result = is_garbled(test)
        print(f"Text: {test[:30]:<30} | Garbled: {result}")

    try:
        # 纯中文
        print(is_language_mix("这是一个测试字符串"))
        # 纯英文
        print(is_language_mix("This is a test string"))
        # 中英混杂，比例适中
        print(is_language_mix("This is 一个 test 字符串"))
        # 中英混杂，比例过高
        print(is_language_mix("This测试is混a杂test"))
    except ValueError as e:
        print(f"错误: {e}")
