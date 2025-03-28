import os
import json
import csv
import requests
import time
import re
from datetime import datetime

# API Configuration
API_URL = "XXXXXXXXX"
API_KEY = "XXXXXXXXXXXXXXXX"
REQUEST_TIMEOUT = 30
RETRY_COUNT = 3
RETRY_DELAY = 5
PROCESSED_FILES_LOG = "/root/eval/processed_files.txt"

EVALUATION_PROMPT = """你是一位对上海市陆家嘴地区人群行为活动有充分了解的，专业的活动链评估专家。请根据以下四个维度对提供的活动链进行0-10分的评估，并按照指定格式输出结果。

## 评分维度及标准

### 1. 时间逻辑一致性 (0-10分)
* **0-2分**: 严重时间冲突，多处时序错误或不可能的时间安排，如在不同地点的活动时间重叠，或者交通时间明显不足
* **3-4分**: 明显时间安排不合理，如工作时长异常，休息不足，用餐时间不符合常规
* **5-6分**: 基本时间顺序正确，但部分活动持续时间不合理或转换时间计算有误
* **7-8分**: 时间安排合理，符合常规生活节奏，少量时间转换不精确但不影响整体时间链
* **9-10分**: 完美时间安排，活动持续时间与转换时间均合理，符合上海都市生活节奏，考虑到高峰期交通等现实因素

### 2. 活动目的连贯性 (0-10分)
* **0-2分**: 活动之间无逻辑关联，随机组合，转换缺乏合理性
* **3-4分**: 活动序列有明显不连贯处，多处转换缺乏动机解释
* **5-6分**: 基本活动目的连贯，但部分转换缺乏充足动机或解释
* **7-8分**: 活动目的连贯，转换合理，大部分活动有明确动机
* **9-10分**: 活动目的完全连贯，每个活动转换都有清晰动机和合理解释，形成流畅自然的行为链

### 3. 人物画像匹配度 (0-10分)
* **0-2分**: 活动与人物基本信息(年龄、职业、收入等)严重不符，无法反映陆家嘴特定人群特征
* **3-4分**: 多个活动与人物画像不匹配，或者活动选择与陆家嘴环境不协调
* **5-6分**: 基本符合人物画像，个别活动选择与特征不吻合，对陆家嘴特色反映一般
* **7-8分**: 大部分活动与人物画像匹配，能较好地体现该人物在陆家嘴区域的特征
* **9-10分**: 所有活动完美匹配人物画像，充分反映其在陆家嘴的生活方式、消费水平和职业特点

### 4. 活动真实性与丰富度 (0-10分)
* **0-2分**: 描述极度简略，基本信息缺失，活动不符合真实生活习惯
* **3-4分**: 描述基础但不充分，缺乏细节，多数活动不符合生活规律
* **5-6分**: 包含基本生活活动和必要信息，但描述平淡，缺乏丰富背景和情感描述
* **7-8分**: 描述详细，包含背景和体验，活动安排符合日常规律，但略显公式化
* **9-10分**: 描述丰富生动，包含感受、动机、环境等多层次信息，完美模拟真实生活，包含必要活动和适当随机性

## 输出格式要求
请严格按照以下结构提供评估结果，必须包含所有4个评分项，每行一个评分，格式为：
- 时间逻辑一致性：X分 （X为0-10的整数）
- 活动目的连贯性：X分
- 人物画像匹配度：X分
- 活动真实性与丰富度：X分

请确保：
1. 每行以"- "开头
2. 使用中文冒号"："而非英文冒号
3. 分数为整数，后面可跟"分"字
4. 不要添加任何额外文字或解释

需要你评估的活动链内容如下：
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
    """从模型响应中提取助手内容"""
    if 'model_response' in json_data:
        response = json_data['model_response']
        # 提取'assistant\n'后的内容
        if 'assistant\n' in response:
            return response.split('assistant\n', 1)[1].strip()
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
    input_folder = '/root/output'
    output_file = '/root/eval/result.csv'
    
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
