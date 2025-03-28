import os
import json
import csv
import requests
import time
import re
from datetime import datetime

# API Configuration
API_URL = "https:XXXXXXXXXXXXXXXX"
API_KEY = "sk-XXXXXXXXXXXXXXXXXX"
REQUEST_TIMEOUT = 30  # 设置请求超时时间为30秒
RETRY_COUNT = 3  # 请求失败时重试次数
RETRY_DELAY = 5  # 重试间隔秒数
PROCESSED_FILES_LOG = "/root/for_eval/processed_files.txt"  # 已处理文件的记录

EVALUATION_PROMPT = """你是一位对上海市陆家嘴地区人群行为活动有深入了解的专业活动链评估专家，擅长识别虚假、杜撰或不符合实际的活动链内容。请严格根据以下四个维度对提供的活动链进行0-10分的评估，特别关注以下问题：时间安排过于规整或不合理、地点经纬度反复使用或与实际情况不符、活动内容明显虚构（如工作人群频繁出现旅游或休闲活动）等。对于存在上述问题的内容，请务必大幅扣分。

## 评分维度及标准

### 1. 时间逻辑一致性 (0-10分)
- 考察各活动的起止时间、持续时长和转换时间是否符合现实生活规律。若存在明显杜撰、过于理想化或计算错误（如活动时间完全规整、转换时间与距离不匹配、时间重叠冲突等），应给予低分。
- 0-2分：存在严重时间冲突、活动时段完全不可能或明显虚构；
- 3-4分：时间安排存在多处漏洞或异常，部分活动时长与实际情况不符；
- 5-6分：整体时间顺序正确，但局部细节不合理或转换时间存在虚假情况；
- 7-8分：时间安排基本合理，但有少量细节稍显理想化；
- 9-10分：时间安排严谨、真实可信，充分考虑了日常生活和交通实际。

### 2. 活动目的连贯性 (0-10分)
- 评估活动之间的逻辑关联与动机合理性。注意检测工作人群是否出现与其身份明显不符的旅游、娱乐或其他非工作相关活动，及活动转换缺乏合理解释的情况。
- 0-2分：活动间毫无关联，转换完全随机或逻辑自相矛盾；
- 3-4分：存在较多不连贯或明显缺乏合理动机的转换；
- 5-6分：基本连贯，但部分活动转换解释不足或与人物身份不符；
- 7-8分：大部分活动目的清晰合理，但仍有轻微不合理之处；
- 9-10分：所有活动转换均逻辑严密、动机明确，完全符合人物身份和实际情况。

### 3. 人物画像匹配度 (0-10分)
- 根据人物基本信息（年龄、职业、收入、家庭构成、交通工具等）判断活动链是否符合其实际生活习惯。若出现与人物身份极不匹配的活动安排（例如普通工作人员安排奢华旅游、频繁休闲娱乐等），应大幅扣分。
- 0-2分：活动内容与人物画像严重脱节，出现极端虚构或不合理的安排；
- 3-4分：多处活动与人物画像不符或存在明显矛盾；
- 5-6分：整体较为匹配，但个别活动明显不符合该人群特征；
- 7-8分：大部分活动与人物画像相符，但存在少数不合理选择；
- 9-10分：所有活动都与人物基本信息高度匹配，真实反映其生活状态和工作特征。

### 4. 活动真实性与丰富度 (0-10分)
- 考察活动链描述的细致程度、信息真实性和背景合理性。重点关注地点经纬度是否存在大量重复或热门虚构地点、活动描述是否缺乏真实感、细节是否过于公式化以及整体是否脱离实际生活。
- 0-2分：描述极为简略或明显杜撰，缺乏任何真实细节；
- 3-4分：信息基础薄弱，细节错误较多且缺乏可信度；
- 5-6分：包含基本活动和必要信息，但描述较平淡，部分数据存在虚假痕迹；
- 7-8分：描述较为丰富，基本符合生活常态，但仍有部分内容不够真实或细节缺失；
- 9-10分：描述详实、生动、真实可信，完美呈现实际生活场景，数据准确无杜撰。

## 输出格式要求
请严格按照以下结构提供评估结果，每行包含一个评分项，格式如下：
- 时间逻辑一致性：X分
- 活动目的连贯性：X分
- 人物画像匹配度：X分
- 活动真实性与丰富度：X分

其中X为0到10之间的整数，请勿添加任何额外文字或解释。

请评估以下活动链内容：
{activity_chain}"""


def ensure_directory_exists(file_path):
    """确保目录存在"""
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def get_processed_files():
    """读取已处理文件列表"""
    if not os.path.exists(PROCESSED_FILES_LOG):
        ensure_directory_exists(PROCESSED_FILES_LOG)
        return set()
    
    with open(PROCESSED_FILES_LOG, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f.readlines())

def mark_as_processed(file_name):
    """将文件标记为已处理"""
    ensure_directory_exists(PROCESSED_FILES_LOG)
    with open(PROCESSED_FILES_LOG, 'a', encoding='utf-8') as f:
        f.write(f"{file_name}\n")
    print(f"Marked {file_name} as processed")

def setup_csv(output_file):
    """设置CSV文件并添加表头"""
    fieldnames = ['id', '时间逻辑一致性', '活动目的连贯性', '人物画像匹配度', '活动真实性与丰富度']
    
    ensure_directory_exists(output_file)
    file_exists = os.path.isfile(output_file)
    
    with open(output_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
    
    return fieldnames

def extract_assistant_content(json_data):
    """从模型响应中提取完整的模型响应内容"""
    if 'model_response' in json_data:
        return json_data['model_response']
    return None

def evaluate_activity_chain(activity_chain, model="gpt-4o-mini-2024-07-18"):
    """发送活动链到API进行评估，包含重试机制"""
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    
    messages = [
        {"role": "system", "content": EVALUATION_PROMPT.format(activity_chain=activity_chain)}
    ]
    
    payload = {
        "messages": messages,
        "model": model,
        "temperature": 0.6,
        "stream": False
    }
    
    for attempt in range(RETRY_COUNT):
        try:
            print(f"\nEvaluating activity chain... (Attempt {attempt+1}/{RETRY_COUNT})")
            response = requests.post(API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                try:
                    response_json = response.json()
                    print(f"Response structure keys: {list(response_json.keys())}")
                    
                    # 标准OpenAI API格式
                    if 'choices' in response_json and len(response_json['choices']) > 0:
                        if 'message' in response_json['choices'][0]:
                            content = response_json['choices'][0]['message']['content']
                            print("Evaluation complete!")
                            print(f"Content: {content}")
                            return content
                    
                    # 替代格式
                    if 'model_response' in response_json:
                        content = response_json['model_response']
                        if 'assistant\n' in content:
                            extracted_content = content.split('assistant\n', 1)[1].strip()
                            print("Evaluation complete!")
                            print(f"Content: {extracted_content}")
                            return extracted_content
                        else:
                            print("Evaluation complete!")
                            print(f"Content: {content}")
                            return content
                    
                    print(f"Unexpected response format. Response contains keys: {list(response_json.keys())}")
                    return ""
                except Exception as e:
                    print(f"Error parsing response: {str(e)}")
                    print(f"Raw response: {response.text}")
                    return response.text  # 如果JSON解析失败，返回原始文本
            else:
                print(f"API returned status code {response.status_code}: {response.text}")
                
        except requests.exceptions.Timeout:
            print(f"Request timed out after {REQUEST_TIMEOUT} seconds")
        except requests.exceptions.RequestException as e:
            print(f"Request error: {str(e)}")
        
        # 如果不是最后一次尝试，等待后重试
        if attempt < RETRY_COUNT - 1:
            retry_wait = RETRY_DELAY * (attempt + 1)  # 渐进式等待时间
            print(f"Waiting {retry_wait} seconds before retrying...")
            time.sleep(retry_wait)
    
    print(f"Failed after {RETRY_COUNT} attempts. Using default scores.")
    return ""  # 所有尝试失败后返回空字符串

def parse_evaluation_scores(evaluation_text):
    """从评估响应中解析分数，增强稳健性"""
    scores = {}
    dimensions = ['时间逻辑一致性', '活动目的连贯性', '人物画像匹配度', '活动真实性与丰富度']
    
    print("Raw evaluation text:")
    print(evaluation_text)
    
    # 检查是否处理的是未正确解析的JSON响应
    if isinstance(evaluation_text, str) and evaluation_text.startswith('{') and '"choices"' in evaluation_text:
        try:
            # 尝试从可能截断的JSON中提取内容
            content_match = re.search(r'"content":"(.*?)","refusal"', evaluation_text)
            if content_match:
                evaluation_text = content_match.group(1).replace('\\n', '\n')
                print(f"Extracted content from JSON: {evaluation_text}")
        except Exception as e:
            print(f"Error extracting from JSON: {str(e)}")
    
    # 逐行尝试解析分数
    for dimension in dimensions:
        score = None
        
        # 在每行中查找维度
        for line in evaluation_text.split('\n'):
            line = line.strip()
            if dimension in line:
                print(f"Found dimension line: {line}")
                
                # 模式1：使用中文冒号 "- dimension：X分"
                pattern1 = f"- {dimension}：(\\d+)分"
                match1 = re.search(pattern1, line)
                if match1:
                    score = int(match1.group(1))
                    print(f"Pattern 1 matched: {score}")
                    break
                
                # 模式2：使用英文冒号 "- dimension: X分"
                pattern2 = f"- {dimension}: (\\d+)分"
                match2 = re.search(pattern2, line)
                if match2:
                    score = int(match2.group(1))
                    print(f"Pattern 2 matched: {score}")
                    break
                
                # 模式3：在行中查找任何数字
                numbers = re.findall(r'\d+', line)
                if numbers:
                    score = int(numbers[0])
                    print(f"Pattern 3 matched: {score}")
                    break
        
        # 如果找到有效分数，添加它
        if score is not None:
            scores[dimension] = score
        else:
            # 如果无法解析分数，使用默认值
            scores[dimension] = 5
            print(f"Warning: Could not parse score for {dimension}, using default value 5")
    
    # 确保分数在0-10范围内
    for dim, score in scores.items():
        if score < 0 or score > 10:
            print(f"Warning: Score for {dim} is outside the valid range (0-10): {score}")
            scores[dim] = max(0, min(10, score))
    
    print(f"Extracted scores: {scores}")
    return scores

def save_results_to_csv(record_id, scores, output_file, fieldnames):
    """将评估结果保存到CSV"""
    row = {'id': record_id}
    row.update(scores)
    
    with open(output_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)
    
    print(f"Results for ID {record_id} saved to CSV")

def process_json_file(file_path, file_name, output_file, fieldnames, processed_files):
    """处理单个JSON文件"""
    # 检查文件是否已处理
    if file_name in processed_files:
        print(f"File {file_name} already processed, skipping...")
        return
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        record_id = data.get('id')
        print(f"\n\nProcessing record ID: {record_id}")
        
        # 提取活动链内容
        activity_chain = extract_assistant_content(data)
        
        if not activity_chain:
            print(f"Warning: Could not extract activity chain from file {file_path}")
            mark_as_processed(file_name)  # 即使提取失败也标记为已处理
            return
        
        # 发送评估
        evaluation_response = evaluate_activity_chain(activity_chain)
        
        # 解析分数
        scores = parse_evaluation_scores(evaluation_response)
        
        # 保存结果
        save_results_to_csv(record_id, scores, output_file, fieldnames)
        
        # 标记为已处理
        mark_as_processed(file_name)
        
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        # 不标记为已处理，以便下次重试

def main():
    """主函数，处理所有JSON文件"""
    # 设置输入和输出路径
    input_folder = '/root/for_eval'
    output_file = '/root/for_eval/result.csv'
    
    # 验证输入文件夹
    if not os.path.exists(input_folder):
        print(f"Error: Folder {input_folder} does not exist.")
        return
    
    # 获取已处理文件列表
    processed_files = get_processed_files()
    print(f"Found {len(processed_files)} already processed files")
    
    # 设置CSV文件
    fieldnames = setup_csv(output_file)
    
    # 列出所有JSON文件
    json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON files found in {input_folder}")
        return
    
    remaining_files = [f for f in json_files if f not in processed_files]
    print(f"Found {len(json_files)} JSON files, {len(remaining_files)} remaining to process")
    
    # 处理每个JSON文件
    for i, file_name in enumerate(json_files):
        if file_name in processed_files:
            continue
            
        file_path = os.path.join(input_folder, file_name)
        print(f"\nProcessing file {i+1}/{len(json_files)}: {file_name}")
        
        try:
            process_json_file(file_path, file_name, output_file, fieldnames, processed_files)
            # 添加小延迟以避免API速率限制
            time.sleep(1)
        except Exception as e:
            print(f"Error in main loop processing file {file_path}: {str(e)}")
    
    print(f"\nAll files processed. Results saved to {output_file}")

if __name__ == "__main__":
    print("Starting Activity Chain Evaluation Script")
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Connecting to gpt-4o-mini-2024-07-18 model via xiaoai.plus API")
    print("-" * 50)
    
    main()
