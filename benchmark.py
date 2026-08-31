#!/usr/bin/env python3
"""
Automated Telemetry & Benchmark Harness for Qwen3.8-Flash-Next on DGX Spark.
Measures TTFT (Time to First Token), Token Generation Throughput, and Concurrency.
"""
import argparse
import asyncio
import time
import json
import statistics
import aiohttp

DEFAULT_PROMPTS = [
    "Write a high-performance concurrent queue in Rust using atomic operations and crossbeam primitives.",
    "Explain the architectural differences between Gated DeltaNet (GDN) and Transformer Self-Attention in detail.",
    "Implement an optimized CUDA kernel for mixed-precision FP4 matrix multiplication with Tensor Cores.",
    "Refactor a distributed microservice architecture from synchronous REST to event-driven Kafka with idempotency guarantees."
]

async def send_prompt(session, url, api_key, model, prompt, max_tokens):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an elite software architect and computer systems expert."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True
    }

    start_time = time.perf_counter()
    ttft = None
    generated_tokens = 0

    async with session.post(f"{url}/v1/chat/completions", headers=headers, json=payload) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise Exception(f"HTTP {resp.status}: {text}")

        async for chunk in resp.content:
            chunk_str = chunk.decode("utf-8")
            for line in chunk_str.split("\n"):
                if line.startswith("data: ") and line.strip() != "data: [DONE]":
                    if ttft is None:
                        ttft = time.perf_counter() - start_time
                    generated_tokens += 1

    total_time = time.perf_counter() - start_time
    gen_time = total_time - (ttft or 0)
    tok_per_sec = generated_tokens / gen_time if gen_time > 0 else 0

    return {
        "ttft": ttft or total_time,
        "total_time": total_time,
        "tokens": generated_tokens,
        "tok_per_sec": tok_per_sec
    }

async def run_benchmark(url, api_key, model, concurrency, num_requests, max_tokens):
    print(f"\n========================================================")
    print(f"  Benchmarking Qwen3.8-Flash-Next on {url}")
    print(f"  Concurrency: {concurrency} | Total Requests: {num_requests}")
    print(f"========================================================\n")

    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for i in range(num_requests):
            prompt = DEFAULT_PROMPTS[i % len(DEFAULT_PROMPTS)]
            tasks.append(send_prompt(session, url, api_key, model, prompt, max_tokens))

        results = await asyncio.gather(*tasks, return_exceptions=True)

    valid_results = [r for r in results if isinstance(r, dict)]
    errors = [r for r in results if not isinstance(r, dict)]

    if not valid_results:
        print(f"Error: All requests failed! Exceptions: {errors[:3]}")
        return

    ttfts = [r["ttft"] * 1000 for r in valid_results]
    speeds = [r["tok_per_sec"] for r in valid_results]
    total_tokens = sum(r["tokens"] for r in valid_results)

    print("\n--- 📊 Telemetry Results ---")
    print(f"Successful Requests: {len(valid_results)} / {num_requests}")
    print(f"Average TTFT:        {statistics.mean(ttfts):.2f} ms (p95: {statistics.quantiles(ttfts, n=20)[18]:.2f} ms)")
    print(f"Avg Decode Speed:    {statistics.mean(speeds):.2f} tokens/s (per stream)")
    print(f"Peak Decode Speed:   {max(speeds):.2f} tokens/s")
    print(f"Total Tokens Output: {total_tokens} tokens")
    print("========================================================\n")

def main():
    parser = argparse.ArgumentParser(description="Benchmark Qwen3.8-Flash-Next on DGX Spark")
    parser.add_argument("--host", default="127.0.0.1", help="Target host")
    parser.add_argument("--port", default=8000, type=int, help="Target port")
    parser.add_argument("--api-key", default="local", help="API key")
    parser.add_argument("--model", default="qwen3.8-flash-next", help="Model name")
    parser.add_argument("--concurrency", default=4, type=int, help="Concurrency level")
    parser.add_argument("--num-requests", default=12, type=int, help="Total requests")
    parser.add_argument("--max-tokens", default=512, type=int, help="Max tokens to generate")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    asyncio.run(run_benchmark(url, args.api_key, args.model, args.concurrency, args.num_requests, args.max_tokens))

if __name__ == "__main__":
    main()
