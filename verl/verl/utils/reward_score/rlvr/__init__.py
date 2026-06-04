#!/usr/bin/env python
# encoding: utf-8
import json
import os
import re
import sys
import time
import traceback
import uuid

import grpc
import numpy as np
from math_verify import parse, verify

cur_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.abspath(os.path.join(cur_path, "proto/gen/python")))


import asyncio
from collections import Counter

import general_servlet_grpc_pb2 as pb2
import general_servlet_grpc_pb2_grpc as pb2_grpc
import pandas as pd
from loguru import logger

logger.configure(
    handlers=[
        {
            "sink": sys.stdout,
            "format": "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>[{process}]</cyan>:<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            "level": "INFO",
        }
    ]
)


TIMEOUT = 10000.0  # seconds.


class AsyncClient:
    def __init__(self, url):
        self.url = url

    async def __aenter__(self):
        self.channel = grpc.aio.insecure_channel(self.url)
        self.stub = pb2_grpc.AgentServiceStub(self.channel)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.channel.close()

    async def send_request(self, payload):
        try:
            metadata = [
                ("x-model", "rlvr-copy"),
                # ("x-model", "minicpm3-4b-1209"),
                ("x-trace-id", uuid.uuid4().__str__()),
            ]
            health_repsonse = await self.stub.Health(
                pb2.HealthCheckRequest(service="health check"),
                timeout=TIMEOUT,
                metadata=metadata,
            )
            if (
                health_repsonse.status
                == pb2.HealthCheckResponse.ServingStatus.NOT_SERVING
            ):
                yield []
            else:
                async for response in self.stub.AgentResStream(
                    pb2.Request(payload=payload.encode("utf-8")),
                    timeout=TIMEOUT,
                    metadata=metadata,
                ):
                    yield response
        except Exception as e:
            logger.error(f"gRPC error: {e}")
            # logger.error(f"gRPC error: {e}, payload: {payload}")

            ret = {
                "token_decode": 0,
                "token_encode": 0,
                "stoped": True,
                "stop_reasons": ["error"],
                "error_code": str(e.code()),
            }
            yield pb2.Response(
                code=500,
                data=json.dumps(ret, ensure_ascii=False).encode("utf-8"),
            )


async def main(cnt, qps, queries):
    # async with AsyncClient("10.17.7.178:31202") as client:
    async with AsyncClient("127.0.0.1:9001") as client:
        begin = time.time()
        rts = []
        token_output = []
        token_input = []
        token_total = []
        error_list = []
        semaphore = asyncio.Semaphore(qps)
        error_counter = Counter()

        async def send_with_delay(query):
            async with semaphore:
                payload = json.dumps(query)
                start = time.time()
                res = await client.send_request(payload)
                rts.append((time.time() - start) * 1000)
                if res is not None:
                    data = json.loads(res.data.decode("utf-8"))

                    error_code = data.get("error_code", "success")
                    error_counter[error_code] += 1
                else:
                    error_list.append(1)

            await asyncio.sleep(1)

        tasks = []
        for i in range(cnt):
            queries[i]["query"]["messages"] = queries[i]["query"]["messages"]
            tasks.append(send_with_delay(queries[i]))
        await asyncio.gather(*tasks)

        dn = (time.time() - begin) * 1000.0
        print(
            f"#Req={cnt} RT={sum(rts) / len(rts):.1f}ms QPS={qps} 时长={dn:.1f}ms"
        )
        print(f"Total throughput: {sum(token_total) * 1000.0 / dn:.1f} token/s")
        print(
            f"Avg prompt throughput: {sum(token_input) * 1000.0 / dn:.1f} token/s"
        )
        print(
            f"Avg out throughput: {sum(token_output) * 1000.0 / dn:.1f} token/s"
        )
        print(f"Error rate: {sum(error_list) * 1.0 / len(error_list)}")
        print(f"Error info", error_counter)

        await asyncio.sleep(10)


async def easy_client(question, label, answer):
    PROMPT = (
        "User: ### Question: {question}\n\n"
        "### Ground Truth Answer: {reference}\n\n"
        "### Student Answer: {response}\n\n"
        "For the above question, please verify if the student's answer is equivalent to the ground truth answer.\n"
        "Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.\n"
        'If the student\'s answer is correct, output "Final Decision: Yes". If the student\'s answer is incorrect, output "Final Decision: No". Assistant:'
    )

    prompt_question = PROMPT.format(
        question=question, reference=label, response=answer
    )

    messages = [
        {"role": "user", "content": prompt_question},
    ]
    payload = {
        "query": {"messages": messages},
        "config": {
            "type": "sample",
            "max_tokens": 1000,
            # "sampling": True,
            # "top_p": 0.8,
            # "top_k": 100,
            "temperature": 0.2,
            # "repetition_penalty": 1.2
            # "num_beams": 2
        },
    }

    # async with AsyncClient("10.17.2.3:80") as client:  # 开发
    final_content = ""
    async with AsyncClient("10.32.181.62:31205") as client:  # 开发
        res = client.send_request(json.dumps(payload, ensure_ascii=False))
        async for cur in res:
            data = json.loads(cur.data)
            # print(data)
            final_content += data["results"][0]["content"]
            if data["stoped"]:
                break
    return final_content


def extract_last_boxed(text: str) -> str:
    """
    Extract the last occurrence of a boxed answer from the input text.

    Returns:
        The content inside the last \boxed{...} or None if not found.
    """
    pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    matches = list(re.finditer(pattern, text))
    if matches:
        return matches[-1].group(1)
    return None


def extract_last_final_answer(text: str) -> str:
    """
    Try to extract the final answer from the text using several candidate patterns.

    Returns:
        The extracted answer as a string, or None if none of the patterns match.
    """
    candidate_patterns = [
        r"Final Answer:\s*((?:[^<]|<[^<])*?)\n",
        r"Final Answer is:\s*((?:[^<]|<[^<])*?)\n",
        r"The answer is:\s*((?:[^<]|<[^<])*?)\n",
        r"Answer:\s*((?:[^<]|<[^<])*?)\n",
        r"Solution:\s*((?:[^<]|<[^<])*?)\n",
        r"The solution is:\s*((?:[^<]|<[^<])*?)\n",
    ]

    last_match = None
    last_position = -1
    for pattern in candidate_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            if match.start() > last_position:
                last_position = match.start()
                last_match = match.group(1).strip()

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
    for stop_word in stop_words:
        if last_match and last_match.endswith(stop_word):
            last_match = last_match[: -len(stop_word)].strip()

    return last_match


def extract_solution(solution_str: str) -> str:
    boxed_answer = extract_last_boxed(solution_str)
    if boxed_answer:
        return boxed_answer
    return extract_last_final_answer(solution_str)


def compute_score(
    solution_str,
    ground_truth,
    extra_info,
    format_reward=0.1,
    answer_reward=1.0,
    retry_times=3,
    debug=False,
):
    try:
        if isinstance(extra_info, str):
            extra_info = json.loads(extra_info)
        if "question" in extra_info and isinstance(extra_info["question"], str):
            question = extra_info["question"]
        elif "prompt" in extra_info and isinstance(extra_info["prompt"], str):
            question = extra_info["prompt"]
        elif "prompt" in extra_info and isinstance(extra_info["prompt"], list):
            question = extra_info["prompt"][-1]["content"]
        else:
            raise ValueError("Question is empty")
        solution_str = extract_solution(solution_str)
        if solution_str is None or len(solution_str) == 0:
            return 0, "Solution is empty"
        try:
            if verify(parse(solution_str), parse(ground_truth)):
                return answer_reward, "math verify pass"
        except Exception as e:
            print(f"Error: {e}, {traceback.format_exc()}")
        final_content = ""
        for _ in range(retry_times):
            try:
                final_content = asyncio.run(
                    easy_client(question, ground_truth, solution_str)
                )
                break
            except Exception as e:
                time.sleep(1)
                print(f"Error: {e}, {traceback.format_exc()}")
                continue
        # print(
        #     f"solution: {solution_str}, ground_truth: {ground_truth}\n final_content: {final_content}"
        # )
        if "Final Decision: Yes" in final_content:
            print("✅")
            return answer_reward, final_content
        else:
            print("❌")
            return 0, final_content
    except Exception as e:
        print(traceback.format_exc())
        return 0, f"Error: {traceback.format_exc()}"


if __name__ == "__main__":
    result = asyncio.run(
        easy_client("What is the result of 2 + 3?", "5", "5.0")
    )
    print(result)
