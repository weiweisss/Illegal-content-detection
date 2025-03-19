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
            reason1=entry["reason1"],
            extended_type=entry["extended_type"],
            reason2=entry["reason2"],
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
        model="deepseek-reasoner",
        concurrency=10
    )

    input_file = 'new.json'
    output_file = 'Cot.json'

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


