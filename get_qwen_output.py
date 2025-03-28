#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
陆家嘴活动轨迹数据生成脚本
用于生成对话数据，保存到指定目录下的JSON文件中
"""

import os
import json
import time
import random
import logging
import traceback
from datetime import datetime
import requests
from tqdm import tqdm  # 进度条显示

# ==========配置部分==========
# 基本配置
OUTPUT_DIR = "/root/for_eval"
NUM_DIALOGUES = 2
API_KEY = "sk-XXXXXXXXXXXXX"  # 硬编码的API密钥
API_BASE = "https:XXXXXXXXXXXXX"
MODEL_NAME = "XXXXXXXXXX"

# 配置日志
LOG_FILE = os.path.join(OUTPUT_DIR, "generation_log.log")
LOG_LEVEL = logging.INFO

# API请求配置
MAX_RETRIES = 3           # 最大重试次数
RETRY_DELAY = 2           # 重试间隔（秒）
REQUEST_TIMEOUT = 60      # 请求超时时间（秒）
RATE_LIMIT_DELAY = 1      # 请求间隔（秒）

# ==========初始化部分==========
# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 配置日志
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("dialogue_generator")

# ==========模板部分==========
# 系统消息模板（包含详细的格式要求）
system_message = """**请基于真实世界信息，生成一个在上海市陆家嘴区域内进行活动的人，在某一典型工作日内的完整活动轨迹信息。要求：时间安排符合上海都市生活作息规律，空间位置限定在上海市陆家嘴区域内，活动轨迹需体现通勤、工作、餐饮、休闲等日常行为，且符合现代都市生活的真实场景与逻辑。坐标信息均为1984坐标系。

请按照以下结构输出信息：

# 此人个体基本信息
\t[陆家嘴活动人群画像]:(类型)
\t[年龄]: XX岁
\t[性别]: (男性/女性)
\t[家庭结构]: (类型)
\t[个人月收入]: XX元/月
\t[家庭可支配收入]: XX元/月
\t[交通工具保有情况]: (交通工具列表)
---
# 在上海市陆家嘴区域内一天内的完整活动轨迹记录
## 出行ID：1[第1次出行]  |  出行方式：(交通方式)
\t[时段]：HH:MM:SS - HH:MM:SS, 出行时耗(XX.X)分钟
\t[起点终点]：起点经纬度(XXX.XXXXXX,YY.YYYYYY)，终点经纬度(XXX.XXXXXX,YY.YYYYYY)
\t[距离]：直线距离(XXXX.XX)米
\t[出行目的]：(目的)
\t[出行类型]：(类型)
\t[交通方式选择动因]：(原因描述)
\t[交通方式体验]：(体验描述)
\t[交通方式转换意愿]:(意愿描述)

## 活动ID：1 | 活动类型：(类型)
\t[时段]：HH:MM:SS-HH:MM:SS,累计(XXX)分钟
\t[地点]：名称为(地点名称),类型为(地点类型),坐标经纬度为(XXX.XXXXXXXX,YY.YYYYYYYYYY)
\t[活动内容]：(详细描述)
\t[时空制约]：(制约描述)
\t[时空灵活度评分]：时间自由度:X（0=严格固定, 10=随时可调整）;空间自由度:X（0=必须特定地点, 10=任意地点）
\t[活动评价]：
\t\t- 活动动机：(动机描述)
\t\t- 决策过程：(决策描述)

(根据人物行程可添加更多出行和活动记录)

备注：在X:XX:XX之前和XX:XX:XX之后，我都不在陆家嘴内部活动。

[活动链出行链概述]：
\t[Ingress Phase]出发到达陆家嘴前活动：从(地点类型)，坐标(XXX.XXXXXXXXXXXXX,YY.YYYYYYYYYYYYYY)出发，然后到达陆家嘴进行上面的活动及出行
\t[Egress Phase]离开陆家嘴后的活动：到达(地点类型)，坐标(XXX.XXXXXXXXXXXXX,YY.YYYYYYYYYYYYYY)，结束在陆家嘴一天内的活动

# 此人在上海市陆家嘴区域内进行上述完整的活动后的主观评价与建议

## 评分指标与说明：
\t[工作效率]：X分，(评价描述)
\t[休闲满意度]：X分，(评价描述)
\t[交通便利度]：X分，(评价描述)
\t[社交互动]：X分，(评价描述)**"""

# 用户提示模板列表
user_prompt_templates = [
    "请生成一份陆家嘴区域内某人的日常活动轨迹信息。", 
    "生成一个在陆家嘴活动的人的一天行程记录。", 
    "描述一位在陆家嘴区域内活动者的全天行程轨迹。", 
    "请给出陆家嘴区域内某人一天的活动轨迹信息。", 
    "生成一份记录陆家嘴区域内某人全天活动的轨迹信息。", 
    "请生成陆家嘴内一位人士的一天活动轨迹。", 
    "生成陆家嘴区域内某人的一日活动轨迹记录。", 
    "描述陆家嘴区域内一位活动者的全天行程。", 
    "请给出陆家嘴区域内某人的全天活动轨迹。", 
    "生成陆家嘴内某人一天的行程信息。", 
    "请生成陆家嘴区域内某人的日常行程记录。", 
    "描述一个在陆家嘴活动的人的全天轨迹。", 
    "生成陆家嘴区域内一位人士全天的活动记录。", 
    "请提供陆家嘴区域内某人一天的详细行程。", 
    "生成一份关于陆家嘴内某人全天活动的轨迹记录。", 
    "请描述陆家嘴区域内一位人士的全天行程。", 
    "生成陆家嘴区域内某人的日常活动轨迹。", 
    "请生成一份记录陆家嘴内某人全天行程的提示信息。", 
    "生成陆家嘴内某人一日活动的轨迹记录。", 
    "请给出陆家嘴区域内某人一天行程的完整轨迹信息。", 
    "Please generate a detailed daily itinerary for a person active in the Lujiazui area.", 
    "Generate an activity log for a person spending their day in Lujiazui, Shanghai.", 
    "Describe the daily journey of someone who works and lives in Lujiazui.", 
    "Provide a full-day activity record for an individual in the Lujiazui district.", 
    "Produce a daily schedule outlining various activities of a person in Lujiazui."
]

# ==========功能函数部分==========
def check_output_file(index):
    """检查输出文件是否已存在"""
    filename = os.path.join(OUTPUT_DIR, f"dialogue_{index}.json")
    return os.path.exists(filename)

def make_api_request(user_prompt, retry_count=0):
    """
    向API发送请求并获取回复
    
    参数:
        user_prompt (str): 用户提示文本
        retry_count (int): 当前重试次数
        
    返回:
        tuple: (成功标志, 回复内容或错误信息)
    """
    if retry_count >= MAX_RETRIES:
        return False, f"超过最大重试次数 {MAX_RETRIES}"
    
    try:
        # 准备请求头
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
        
        # 准备请求数据
        data = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ]
        }
        
        # 发送请求
        logger.debug(f"发送API请求: {user_prompt[:30]}...")
        response = requests.post(
            f"{API_BASE}/chat/completions",
            headers=headers,
            json=data,
            timeout=REQUEST_TIMEOUT
        )
        
        # 检查HTTP状态码
        if response.status_code != 200:
            error_msg = f"API错误: 状态码:{response.status_code}, 响应:{response.text}"
            logger.error(error_msg)
            
            # 对特定错误码进行处理
            if response.status_code == 429:  # 请求频率限制
                logger.warning(f"请求频率限制，等待{RETRY_DELAY*2}秒后重试...")
                time.sleep(RETRY_DELAY * 2)  # 等待更长时间
                return make_api_request(user_prompt, retry_count + 1)
            elif response.status_code >= 500:  # 服务器错误
                logger.warning(f"服务器错误，等待{RETRY_DELAY}秒后重试...")
                time.sleep(RETRY_DELAY)
                return make_api_request(user_prompt, retry_count + 1)
            
            return False, error_msg
        
        # 解析响应
        result = response.json()
        
        # 提取助手回复文本
        if "choices" in result and len(result["choices"]) > 0:
            assistant_response = result["choices"][0]["message"]["content"]
            return True, assistant_response
        else:
            error_msg = f"无效的API响应格式: {result}"
            logger.error(error_msg)
            return False, error_msg
            
    except requests.exceptions.Timeout:
        logger.warning(f"请求超时，等待{RETRY_DELAY}秒后重试...")
        time.sleep(RETRY_DELAY)
        return make_api_request(user_prompt, retry_count + 1)
        
    except requests.exceptions.ConnectionError:
        logger.warning(f"连接错误，等待{RETRY_DELAY}秒后重试...")
        time.sleep(RETRY_DELAY)
        return make_api_request(user_prompt, retry_count + 1)
        
    except Exception as e:
        error_msg = f"请求异常: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        time.sleep(RETRY_DELAY)
        return make_api_request(user_prompt, retry_count + 1)

def validate_response(response_text):
    """
    验证模型回复是否符合预期格式
    
    参数:
        response_text (str): 模型回复文本
        
    返回:
        bool: 是否有效
    """
    # 检查基本关键部分是否存在
    required_sections = [
        "# 此人个体基本信息",
        "# 在上海市陆家嘴区域内一天内的完整活动轨迹记录",
        "## 出行ID：",
        "## 活动ID："
    ]
    
    for section in required_sections:
        if section not in response_text:
            logger.warning(f"回复缺少必要部分: {section}")
            return False
    
    # 检查基础信息字段是否完整
    basic_info_fields = [
        "[陆家嘴活动人群画像]",
        "[年龄]",
        "[性别]",
        "[家庭结构]",
        "[个人月收入]",
        "[家庭可支配收入]",
        "[交通工具保有情况]"
    ]
    
    for field in basic_info_fields:
        if field not in response_text:
            logger.warning(f"回复缺少基本信息字段: {field}")
            return False
    
    return True

def generate_dialogue(index):
    """
    生成单个对话并保存到JSON文件
    
    参数:
        index (int): 对话序号
        
    返回:
        bool: 是否成功
    """
    # 如果文件已存在，跳过生成
    if check_output_file(index):
        logger.info(f"对话 {index} 文件已存在，跳过生成")
        return True
    
    # 随机选择一个用户提示
    user_prompt = random.choice(user_prompt_templates)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        # 记录开始时间
        start_time = time.time()
        
        # 发送API请求
        success, response = make_api_request(user_prompt)
        
        if not success:
            logger.error(f"生成对话 {index} 失败: {response}")
            return False
        
        # 验证响应格式
        if not validate_response(response):
            logger.warning(f"对话 {index} 响应格式验证失败，将重试")
            return False
        
        # 计算生成时间
        generation_time = time.time() - start_time
        
        # 格式化完整模型响应
        model_response = f"system\n{system_message}\nuser\n{user_prompt}\nassistant\n{response}"
        
        # 创建对话对象
        dialogue = {
            "id": index,
            "timestamp": timestamp,
            "system_message": system_message,
            "user_prompt": user_prompt,
            "model_response": model_response,
            "generation_time": generation_time
        }
        
        # 保存到文件
        filename = os.path.join(OUTPUT_DIR, f"dialogue_{index}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dialogue, f, ensure_ascii=False, indent=2)
        
        logger.info(f"成功生成对话 {index}: 提示={user_prompt[:20]}..., 时间={generation_time:.2f}秒")
        return True
    
    except Exception as e:
        logger.error(f"生成对话 {index} 时发生异常: {str(e)}\n{traceback.format_exc()}")
        return False

def main():
    """主函数：生成所有对话"""
    logger.info(f"开始生成 {NUM_DIALOGUES} 个对话，输出目录: {OUTPUT_DIR}")
    logger.info(f"使用模型: {MODEL_NAME}")
    
    # 检查API密钥
    if not API_KEY or API_KEY == "your_api_key_here":
        logger.error("API密钥未设置或无效")
        return
    
    # 显示已有文件数量
    existing_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('dialogue_') and f.endswith('.json')]
    logger.info(f"输出目录中已有 {len(existing_files)} 个对话文件")
    
    successful = 0
    index = 1
    
    # 使用进度条显示处理进度
    with tqdm(total=NUM_DIALOGUES, desc="生成对话") as pbar:
        while successful < NUM_DIALOGUES:
            # 尝试生成对话
            if generate_dialogue(index):
                successful += 1
                pbar.update(1)
                index += 1
            else:
                logger.warning(f"对话 {index} 生成失败，将重试...")
                time.sleep(RETRY_DELAY)
            
            # 为避免请求频率限制，添加延迟
            time.sleep(RATE_LIMIT_DELAY)
    
    # 验证生成的文件数量
    final_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('dialogue_') and f.endswith('.json')]
    logger.info(f"成功生成 {successful} 个对话，输出目录中共有 {len(final_files)} 个对话文件")
    
    # 输出一些文件示例
    if final_files:
        sample_files = sorted(final_files)[:5]  # 取前5个文件作为示例
        logger.info(f"示例文件: {', '.join(sample_files)}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("用户中断，程序已停止")
    except Exception as e:
        logger.critical(f"程序执行过程中发生未处理异常: {str(e)}\n{traceback.format_exc()}")
