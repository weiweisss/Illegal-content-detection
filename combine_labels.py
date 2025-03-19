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
aclient = AsyncOpenAI(
    api_key="sk-141733f5b343421492093217fd493e5f",
    base_url="https://api.deepseek.com"
)

# 新增固定输出列
OUTPUT_COLUMNS = ['context', 'original_label', 'extended_type', 'reason', 'api_response']


async def call_deepseek_api(session, context, original_label, model, sem):
    async with sem:
        task = (
            "Based on the text (context) and the known primary violation type (original_label), analyze whether the text may involve other potential violation types. "
            "The candidate list of extended types includes: moral, race, socioeconomic, gender, disability, nationality, sexual-orientation, physical-appearance, religion, age, profession, rumor, insult, hate."
            "\nEXAMPLE INPUT:\n"
            "context: This is offensive content\n"
            "original_label: hate\n"
            "\nEXAMPLE JSON OUTPUT:\n"
            "{\n"
            '    "extended_types": ["insult", "moral"],\n'
            '    "reason": "The content contains both insulting language and violates moral standards"\n'
            "}\n"
            "If no other violation types are found, return an empty array for extended_types."
        )
        messages = [
            {"role": "system", "content": "As a professional content security auditor, you need to judge the types of text violations from multiple dimensions." + task},
            {"role": "user", "content": f"context: {context},original_label: {original_label}"}
        ]
        try:
            response = await aclient.chat.completions.create(
                model=model,
                messages=messages,
                response_format={
                    'type': 'json_object'
                },
                stream=False
            )
            response_text = response.choices[0].message.content.strip()
            # 解析 JSON 响应
            try:
                response_json = json.loads(response_text)
                extended_types = response_json.get('extended_types', [])
                reason = response_json.get('reason', '')
                cleaned_types = ','.join(extended_types) if extended_types else ''
            except json.JSONDecodeError:
                cleaned_types = None
                reason = None
                
            result = {
                'context': context,
                'original_label': original_label,
                'extended_type': cleaned_types if cleaned_types else None,
                'reason': reason,
                'api_response': response_text
            }
            return result
        except requests.exceptions.RequestException as e:
            print(f"API请求失败: {e}")
            return {
                'context': context,
                'original_label': original_label,
                'extended_type': None,
                'reason': None,
                'api_response': str(e)
            }


async def process_batch(batch_df, model, concurrency=10):
    sem = asyncio.Semaphore(concurrency)
    tasks = []

    async with aiohttp.ClientSession() as session:
        for _, row in batch_df.iterrows():
            task = call_deepseek_api(session, row['context'], row['bias_type'],model, sem)
            tasks.append(task)

        responses = []
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing batch"):
            responses.append(await f)

        # 将响应转换为字典列表
        return responses


async def main():
    parser = argparse.ArgumentParser(description='Process data and analyze bias')
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--model', default="deepseek-chat", choices=["deepseek-reasoner", "deepseek-chat"])
    parser.add_argument('--concurrency', default=10, type=int)
    args = parser.parse_args()

    input_file = 'merged.csv'
    output_file = 'label_extended.json'

    df = pd.read_csv(input_file)
    df = df.sample(n=min(10, len(df)), random_state=1337)
    total_rows = len(df)

    all_results = []

    for start_idx in range(0, total_rows, args.batch_size):
        end_idx = min(start_idx + args.batch_size, total_rows)
        batch = df.iloc[start_idx:end_idx]

        print(f"\nProcessing batch {start_idx // args.batch_size + 1}/{(total_rows - 1) // args.batch_size + 1}")
        start_time = time.time()
        batch_results = await process_batch(batch, args.model, args.concurrency)
        end_time = time.time()
        print(f"\nTotal time: {end_time - start_time:.2f} seconds")
        
        all_results.extend(batch_results)
        print(f"Processed {len(batch_results)} results")
    
    # 保存所有结果到JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"All results saved to {output_file}")



if __name__ == "__main__":
    asyncio.run(main())