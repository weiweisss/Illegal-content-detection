import pandas as pd
import asyncio
import aiohttp
from tqdm import tqdm
from openai import AsyncOpenAI
import argparse
import time
import requests
import re
aclient = AsyncOpenAI(
    api_key="xxxx",
    base_url="https://api.deepseek.com"
)

# 新增固定输出列
OUTPUT_COLUMNS = ['context','original_label','extended_type', 'api_response']


async def call_deepseek_api(session, context,original_label, model, sem):
    async with sem:
        task = (
            "Based on the text (context) provided by the user and the known primary violation type (original_label), analyze whether the text may also involve other potential violation types. "
            "The candidate list of extended types includes:moral, race, socioeconomic, gender, disability, nationality, sexual-orientation, physical-appearance, religion, age, profession, rumor, insult, hate."
            "\nObey the following answer format:\n"
            "Other possible violation types: {'type1','type2',...} Reason: {}."
            "\nIf the content does not violate any other type, fill in the Other possible violation types with blank."
        )
        messages = [
            {"role": "system", "content": "As a professional content security auditor, you need to judge the types of text violations from multiple dimensions." + task},
            {"role": "user", "content": f"context: {context},original_label: {original_label}"}
        ]
        try:

            response = await aclient.chat.completions.create(
                model=model,
                messages=messages,
                stream=False
            )
            response_text = response.choices[0].message.content.strip().lower()
            extended_type = response_text.split('reason')[0]
            extended_type = extended_type.split('other possible violation types:')[-1].strip()
            cleaned_types = re.sub(r"[{\'\"}\s]", "", extended_type)
            cleaned_types = re.sub(r",+", ",", cleaned_types).strip(',')
            # 统一返回结构
            result = {
                'context': context,
                'original_label':original_label,
                'extended_type': cleaned_types if cleaned_types else None,
                'api_response': response_text
            }
            return result
        except requests.exceptions.RequestException as e:
            print(f"API请求失败: {e}")
            return {
                'context': context,
                'original_label':original_label,
                'extended_type': None,
                'api_response': str(e)
            }


async def process_batch(batch_df, model, concurrency=10):
    sem = asyncio.Semaphore(concurrency)
    tasks = []

    async with aiohttp.ClientSession() as session:
        for _, row in batch_df.iterrows():
            task = call_deepseek_api(session, row['context'], row['violate_type'],model, sem)
            tasks.append(task)

        responses = []
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing batch"):
            responses.append(await f)

        # 确保列顺序和完整性
        return pd.DataFrame(responses)[OUTPUT_COLUMNS]


async def main():
    parser = argparse.ArgumentParser(description='Process data and analyze bias')
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--model', default="deepseek-chat", choices=["deepseek-reasoner", "deepseek-chat"])
    parser.add_argument('--concurrency', default=10, type=int)
    args = parser.parse_args()

    input_file = 'original.csv'
    output_file = 'label_extended.csv'

    df = pd.read_csv(input_file)
    total_rows = len(df)

    # 初始化带完整表头的文件
    pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(output_file, index=False)

    for start_idx in range(0, total_rows, args.batch_size):
        end_idx = min(start_idx + args.batch_size, total_rows)
        batch = df.iloc[start_idx:end_idx]

        print(f"\nProcessing batch {start_idx // args.batch_size + 1}/{(total_rows - 1) // args.batch_size + 1}")
        start_time = time.time()
        batch_results = await process_batch(batch, args.model, args.concurrency)
        end_time = time.time()
        print(f"\nTotal time: {end_time - start_time:.2f} seconds")
        # 追加数据（不写表头）
        batch_results.to_csv(output_file, mode='a', header=False, index=False)
        print(f"Saved {len(batch_results)} results")


if __name__ == "__main__":
    asyncio.run(main())
