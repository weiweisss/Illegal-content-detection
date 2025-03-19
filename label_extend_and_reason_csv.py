# %%
import pandas as pd
import asyncio
import aiohttp
from tqdm import tqdm
from openai import AsyncOpenAI
import argparse
import time
import requests
import re
import json

# %%
aclient = AsyncOpenAI(
    api_key="sk-141733f5b343421492093217fd493e5f",
    base_url="https://api.deepseek.com"
)



# %%
async def call_deepseek_api(session, context, bias_type, model, sem):
    """调用 Deepseek API 分析文本"""
    async with sem:
        task = (
            "I will provide you with a passage (context) and its violation risk type (bias_type). "
            "Please explain the reason for its violation and determine whether this passage also falls under other violation risk types from the list."
            "Analyze and provide the reasons. If the text is of the 'safe' type, please explain why the content is risk-free."
            " The candidate list of extended types includes: moral, race, socioeconomic, gender, disability, nationality, sexual-orientation, physical-appearance, religion, age, profession, rumor, insult, hate."
            "Strictly follow this JSON format:\n"
            "{\n"
            '  "bias_type": "type",\n'
            '  "reason1": "explanation",\n'
            '  "extended_type": ["type1", "type2"],\n'
            '  "reason2": "explanation"\n'
            "}"
            "\nIf the bias_type of the content is 'safe', fill in the Other possible violation types and reason2 with blank."
            "\nIf the content does not violate any other type, fill in the Other possible violation types with blank."
        )
        messages = [
            {"role": "system", "content": "As a professional content security auditor, you need to judge the types of text violations from multiple dimensions." + task},
            {"role": "user", "content": f"context: {context}, bias_type: {bias_type}"}
        ]
        try:
            response = await aclient.chat.completions.create(
                model=model,
                messages=messages,
                response_format={'type': 'json_object'}
            )
            # 直接解析 JSON 响应
            response_json = json.loads(response.choices[0].message.content)
            # 统一返回结构
            return {
                'context': context,
                'bias_type': bias_type,
                'reason1': response_json.get("reason1", ""),
                'extended_type': response_json.get("extended_type", []),
                'reason2': response_json.get("reason2", "")
            }
        except Exception as e:
            print(f"API请求失败: {e}")
            return {
                "context": context,
                "bias_type": bias_type,
                "reason1": str(e),
                "extended_type": [],
                "reason2": ""
            }

# %%
async def process_batch(json_data, model, concurrency=10):
    """处理数据批次"""
    sem = asyncio.Semaphore(concurrency)
    tasks = []

    # 提前过滤无效数据并生成任务
    valid_entries = []
    for idx, entry in enumerate(json_data):
        if not all(key in entry for key in ["context", "bias_type"]):
            print(f"Skipping invalid entry at index {idx}: {entry}")
            continue
        valid_entries.append(entry)
        task = call_deepseek_api(
            session=None,  # 无需 session 参数
            context=entry["context"],
            bias_type=entry["bias_type"],
            model=model,
            sem=sem
        )
        tasks.append(task)

    # 执行异步任务（保持原始顺序）
    results = []
    with tqdm(total=len(tasks), desc=f"Processing {model}") as pbar:
        for future in asyncio.as_completed(tasks):
            try:
                result = await future
                results.append(result)
                pbar.update(1)
            except Exception as e:
                print(f"Task failed: {str(e)}")

    return results

# %%
async def main():
    args = argparse.Namespace(
        batch_size=10,
        model="deepseek-chat",
        concurrency=10
    )

    input_file = 'merged_test.json'
    output_file = 'new.json'

    # 读取 JSON 文件（已知是标准数组格式）
    df = pd.read_json(input_file, orient='records')

    # 初始化输出文件（确保编码正确）
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump([], f, indent=2, ensure_ascii=False)

    total_rows = len(df)

    # 分批处理
    for start_idx in range(0, total_rows, args.batch_size):
        end_idx = min(start_idx + args.batch_size, total_rows)
        batch = df.iloc[start_idx:end_idx].to_dict(orient='records')

        print(f"\nProcessing batch {start_idx // args.batch_size + 1}/{(total_rows - 1) // args.batch_size + 1}")
        start_time = time.time()
        
        batch_results = await process_batch(batch, args.model, args.concurrency)
        
        end_time = time.time()
        print(f"Batch processed in {end_time - start_time:.2f}s")

        # 追加数据到 JSON
        if batch_results:
            with open(output_file, 'r+', encoding='utf-8') as f:
                existing_data = json.load(f)
                existing_data.extend(batch_results)
                f.seek(0)
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
                f.truncate()  # 确保文件尺寸正确


# %%
if __name__ == "__main__":
    asyncio.run(main())


