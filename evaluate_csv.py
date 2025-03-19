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

# %%
aclient = AsyncOpenAI(
    api_key="sk-141733f5b343421492093217fd493e5f",
    base_url="https://api.deepseek.com"
)

# 新增固定输出列
OUTPUT_COLUMNS = ['context','bias_type','reason1','extended_type','reason2','thought_process','CoT','score']

# %%
async def call_deepseek_api(session, context,bias_type,reason1,extended_type,reason2, model, sem):
    async with sem:
        task = (
            "Please evaluate the accuracy of the following analysis process and provide detailed feedback along with an evaluation score(out of 100)\n"
            "Input Data:\nContent: (context)\nViolation Label:(bias_type) Analysis Process:(reason1)\nOther Extended Violation Labels: (extended_label) Analysis Process:(reason2)"
            "Evaluation Criteria:\n1.Logical Consistency: Is the analysis process logically coherent and methodologically sound?\n" 
            "2.Evidence Support: Are the arguments and evidence cited during the analysis sufficient and relevant?\n"
            "3.Conclusion Accuracy: Do the conclusions align with the provided violation labels?\n"
            "4.Completeness: Does the analysis comprehensively consider all relevant factors?\n"
            "Additional Notes:Ensure the final score reflects the overall quality of the evaluation."
            "\nObey the following answer format strictly:\n"
            "score:{} reason:{}"
            
        )
        messages = [
            {"role": "system", "content": "As a specialized content security auditor, you need to evaluate the quality of text violation judgment and its reasons from multiple dimensions." + task},
            {"role": "user", "content": f"context: {context},bias_type: {bias_type},reason1: {reason1},extended_label:{extended_type},reason2:{reason2}"}
        ]
        try:

            response = await aclient.chat.completions.create(
                model=model,
                messages=messages,
                stream=False
            )
            response_text=response.choices[0].message.content.replace('\n', ' ')
            thought_process=(response_text.split('reason:',1))[1]
            score=((response_text.split('reason:',1))[0]).replace('score:','')
            CoT= response.choices[0].message.reasoning_content.replace('\n', ' ')
           # score= (thought_process.split('###',2))[1]
            # 统一返回结构
            result = {
                'context': context,
                'bias_type':bias_type,
                'reason1':reason1,
                'extended_type': extended_type,
                'reason2': reason2,
                'thought_process':thought_process,
                'CoT':CoT,
                'score':score

            }
            return result
        except requests.exceptions.RequestException as e:
            print(f"API请求失败: {e}")
            return {
                'context': context,
                'bias_type':bias_type,
                'reason1':str(e),
                'extended_type': None,
                'reason2': None,
                'thought_process':None,
                'CoT':None,
                'score':None
            }


# %%
async def process_batch(batch_df, model, concurrency=10):
    sem = asyncio.Semaphore(concurrency)
    tasks = []

    async with aiohttp.ClientSession() as session:
        for _, row in batch_df.iterrows():
            task = call_deepseek_api(session, row['context'], row['bias_type'],row['reason1'],row['extended_type'],row['reason2'],model, sem)
            tasks.append(task)

        responses = []
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing batch"):
            responses.append(await f)

        # 确保列顺序和完整性
        return pd.DataFrame(responses)[OUTPUT_COLUMNS]

# %%
async def main():
    parser = argparse.ArgumentParser(description='Process data and analyze bias')
    parser.add_argument('--batch_size', default=10,type=int)
    parser.add_argument('--model', default="deepseek-reasoner", choices=["deepseek-reasoner", "deepseek-chat"])
    parser.add_argument('--concurrency', default=10, type=int)
    args = parser.parse_args([])

    input_file = 'new.csv'
    output_file = 'cot.csv'

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


# %%
if __name__ == "__main__":
    asyncio.run(await main())


