#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
陆家嘴活动轨迹数据生成脚本
用于生成对话数据，保存到指定目录下的JSON文件中
基于通义千问2.5-7B模型
"""

import os
import json
import time
import random
import logging
import traceback
from datetime import datetime
import requests
import hashlib
import uuid

try:
    from tqdm import tqdm  # 进度条显示
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("提示: 未安装tqdm库，将不显示进度条。可以通过 'pip install tqdm' 安装。")

# ==========配置部分==========
# 基本配置
OUTPUT_DIR = "/root/for_eval"
NUM_DIALOGUES = 100  # 设置要生成的对话数量
API_KEY = "sk-XXXXXXXXXXX"  # API密钥
API_BASE = "https:XXXXXXXXXXXX"
MODEL_NAME = "qwen2.5-7b-instruct"  # 官方文档中的模型名称

# 配置日志
LOG_FILE = os.path.join(OUTPUT_DIR, "generation_log.log")
LOG_LEVEL = logging.INFO

# API请求配置
MAX_RETRIES = 5           # 增加最大重试次数
RETRY_DELAY = 5           # 增加重试间隔（秒）
REQUEST_TIMEOUT = 180     # 增加请求超时时间（秒）
RATE_LIMIT_DELAY = 2      # 增加请求间隔（秒）
PROGRESSIVE_RETRY = True  # 启用渐进式重试延迟
MAX_CONCURRENT_REQUESTS = 3  # 最大并发请求数 (如果使用并行处理)

# ==========初始化部分==========
# 确保输出目录存在
try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"输出目录确认: {OUTPUT_DIR}")
except Exception as e:
    print(f"创建输出目录时出错: {str(e)}")
    exit(1)

# 配置日志
try:
    logging.basicConfig(
        level=LOG_LEVEL,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("dialogue_generator")
    logger.info("日志系统初始化成功")
except Exception as e:
    print(f"初始化日志系统时出错: {str(e)}")
    # 继续执行但无日志记录

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

def calculate_retry_delay(retry_count):
    """计算渐进式重试延迟时间"""
    if PROGRESSIVE_RETRY:
        # 指数级增加重试间隔 (2^retry_count * 基础延迟)
        return min(RETRY_DELAY * (2 ** retry_count), 60)  # 最大延迟60秒
    return RETRY_DELAY

def get_request_id():
    """生成唯一的请求ID"""
    return str(uuid.uuid4())

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
    
    request_id = get_request_id()
    retry_delay = calculate_retry_delay(retry_count)
    
    try:
        # 准备请求头
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
            "X-Request-ID": request_id,
            "Accept": "application/json"
        }
        
        # 准备请求数据 - 按照官方文档设置格式
        data = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
            "top_p": 0.8,
            "max_tokens": 4096,  # 增加最大输出tokens
            "stream": False  # 不使用流式输出
        }
        
        # 发送请求
        logger.debug(f"发送API请求: {user_prompt[:30]}..., 请求ID: {request_id}")
        print(f"发送请求中... (提示: {user_prompt[:30]}..., 请求ID: {request_id})")
        
        response = requests.post(
            f"{API_BASE}/chat/completions",
            headers=headers,
            json=data,
            timeout=REQUEST_TIMEOUT
        )
        
        # 检查HTTP状态码
        if response.status_code != 200:
            error_msg = f"API错误: 状态码:{response.status_code}, 响应:{response.text}, 请求ID:{request_id}"
            logger.error(error_msg)
            print(f"请求失败: 状态码 {response.status_code}, 请求ID: {request_id}")
            
            # 对特定错误码进行处理
            if response.status_code == 429:  # 请求频率限制
                logger.warning(f"请求频率限制，等待{retry_delay*2}秒后重试...")
                print(f"请求频率限制，等待{retry_delay*2}秒后重试...")
                time.sleep(retry_delay * 2)  # 等待更长时间
                return make_api_request(user_prompt, retry_count + 1)
            elif response.status_code == 401:  # 认证失败
                logger.error("API认证失败，请检查API密钥是否正确")
                print("API认证失败，请检查API密钥是否正确")
                return False, "API认证失败，请检查API密钥是否正确"
            elif response.status_code == 400:  # 请求参数错误
                error_data = response.json() if response.text else {}
                error_message = error_data.get('error', {}).get('message', '未知错误')
                logger.error(f"请求参数错误: {error_message}")
                print(f"请求参数错误: {error_message}")
                # 如果是模型相关错误，可能需要修正模型名称
                if "model" in error_message.lower():
                    logger.error("可能是模型名称错误，请检查官方文档确认正确的模型名称")
                    print("可能是模型名称错误，请检查官方文档确认正确的模型名称")
                return False, f"请求参数错误: {error_message}"
            elif response.status_code >= 500:  # 服务器错误
                logger.warning(f"服务器错误，等待{retry_delay}秒后重试...")
                print(f"服务器错误，等待{retry_delay}秒后重试...")
                time.sleep(retry_delay)
                return make_api_request(user_prompt, retry_count + 1)
            
            # 其他错误，等待后重试
            time.sleep(retry_delay)
            return make_api_request(user_prompt, retry_count + 1)
        
        # 解析响应
        try:
            result = response.json()
        except json.JSONDecodeError:
            logger.error(f"响应不是有效的JSON格式: {response.text[:200]}...")
            print(f"响应不是有效的JSON格式")
            time.sleep(retry_delay)
            return make_api_request(user_prompt, retry_count + 1)
        
        # 提取助手回复文本
        if "choices" in result and len(result["choices"]) > 0:
            assistant_response = result["choices"][0]["message"]["content"]
            logger.debug(f"收到API回复: {len(assistant_response)} 字符，请求ID: {request_id}")
            print(f"收到回复: {len(assistant_response)} 字符，请求ID: {request_id}")
            return True, assistant_response
        else:
            error_msg = f"无效的API响应格式: {result}, 请求ID: {request_id}"
            logger.error(error_msg)
            print(f"收到无效的响应格式，请求ID: {request_id}")
            time.sleep(retry_delay)
            return make_api_request(user_prompt, retry_count + 1)
            
    except requests.exceptions.Timeout:
        logger.warning(f"请求超时，等待{retry_delay}秒后重试... 请求ID: {request_id}")
        print(f"请求超时，等待{retry_delay}秒后重试... 请求ID: {request_id}")
        time.sleep(retry_delay)
        return make_api_request(user_prompt, retry_count + 1)
        
    except requests.exceptions.ConnectionError:
        logger.warning(f"连接错误，等待{retry_delay}秒后重试... 请求ID: {request_id}")
        print(f"连接错误，等待{retry_delay}秒后重试... 请求ID: {request_id}")
        time.sleep(retry_delay)
        return make_api_request(user_prompt, retry_count + 1)
        
    except Exception as e:
        error_msg = f"请求异常: {str(e)}, 请求ID: {request_id}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        print(f"请求出现异常: {str(e)}, 请求ID: {request_id}")
        time.sleep(retry_delay)
        return make_api_request(user_prompt, retry_count + 1)

def save_dialogue_to_file(index, dialogue):
    """保存对话到文件，有错误重试几次"""
    filename = os.path.join(OUTPUT_DIR, f"dialogue_{index}.json")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(dialogue, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"保存对话到文件失败 (尝试 {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(1)  # 短暂延迟后重试
            else:
                logger.error(f"无法保存对话 {index} 到文件")
                return False

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
        print(f"对话 {index} 文件已存在，跳过生成")
        return True
    
    # 随机选择一个用户提示
    user_prompt = random.choice(user_prompt_templates)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        # 记录开始时间
        start_time = time.time()
        
        # 发送API请求
        print(f"\n--- 开始生成对话 {index} ---")
        success, response = make_api_request(user_prompt)
        
        if not success:
            logger.error(f"生成对话 {index} 失败: {response}")
            print(f"生成对话 {index} 失败: {response[:100]}...")
            return False
        
        # 计算生成时间
        generation_time = time.time() - start_time
        
        # 提取标识符，用于追踪和错误分析
        dialogue_hash = hashlib.md5(f"{user_prompt}_{timestamp}".encode()).hexdigest()[:8]
        
        # 创建对话对象
        dialogue = {
            "id": index,
            "timestamp": timestamp,
            "system_message": system_message,
            "user_prompt": user_prompt,
            "model_response": response,
            "generation_time": generation_time,
            "dialogue_hash": dialogue_hash,
            "model": MODEL_NAME
        }
        
        # 保存到文件
        if save_dialogue_to_file(index, dialogue):
            logger.info(f"成功生成对话 {index}: 提示={user_prompt[:20]}..., 时间={generation_time:.2f}秒, 哈希={dialogue_hash}")
            print(f"成功生成对话 {index}: 用时 {generation_time:.2f} 秒, 哈希 {dialogue_hash}")
            return True
        else:
            logger.error(f"生成对话 {index} 成功，但保存失败")
            print(f"生成对话 {index} 成功，但保存失败")
            return False
    
    except Exception as e:
        logger.error(f"生成对话 {index} 时发生异常: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"生成对话 {index} 时发生异常: {str(e)}")
        return False

def test_api_connection():
    """测试API连接是否正常工作"""
    print("正在测试API连接...")
    
    try:
        # 使用简单的提示进行测试
        test_prompt = "你好，这是一个API连接测试。"
        
        # 准备请求头
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
        
        # 准备请求数据 - 最小化请求内容以加快测试
        data = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "user", "content": test_prompt}
            ],
            "max_tokens": 50
        }
        
        # 发送请求，较短的超时
        response = requests.post(
            f"{API_BASE}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        # 检查响应
        if response.status_code == 200:
            print("✅ API连接测试成功！可以开始生成对话。")
            logger.info("API连接测试成功")
            return True
        else:
            error_data = {}
            try:
                error_data = response.json()
            except:
                pass
                
            error_message = error_data.get('error', {}).get('message', '未知错误')
            print(f"❌ API连接测试失败: 状态码 {response.status_code}, 错误: {error_message}")
            logger.error(f"API连接测试失败: 状态码 {response.status_code}, 错误: {error_message}")
            
            # 针对不同错误给出建议
            if response.status_code == 401:
                print("建议: 请检查API密钥是否正确。")
            elif response.status_code == 404:
                print("建议: 请检查API端点URL是否正确。")
            elif response.status_code == 400 and "model" in str(error_message).lower():
                print("建议: 请检查模型名称是否正确。您可以参考官方文档: https://bailian.console.aliyun.com/")
                
            return False
            
    except requests.exceptions.Timeout:
        print("❌ API连接测试超时。请检查网络连接和API端点可用性。")
        logger.error("API连接测试超时")
        return False
        
    except requests.exceptions.ConnectionError:
        print("❌ API连接错误。请检查网络连接是否正常。")
        logger.error("API连接测试连接错误")
        return False
        
    except Exception as e:
        print(f"❌ API连接测试异常: {str(e)}")
        logger.error(f"API连接测试异常: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def check_api_model_availability():
    """检查指定的模型是否可用"""
    print(f"正在检查模型 {MODEL_NAME} 是否可用...")
    
    try:
        # 准备请求头
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
        
        # 发送简单请求，只检查模型是否可用
        data = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "user", "content": "测试"}
            ],
            "max_tokens": 5
        }
        
        # 发送请求
        response = requests.post(
            f"{API_BASE}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        # 检查响应
        if response.status_code == 200:
            print(f"✅ 模型 {MODEL_NAME} 可用！")
            logger.info(f"模型 {MODEL_NAME} 可用")
            return True
        else:
            error_data = {}
            try:
                error_data = response.json()
            except:
                pass
                
            error_message = error_data.get('error', {}).get('message', '未知错误')
            print(f"❌ 模型 {MODEL_NAME} 不可用: {error_message}")
            logger.error(f"模型 {MODEL_NAME} 不可用: {error_message}")
            
            # 如果错误信息包含"model"，可能是模型名称错误
            if "model" in str(error_message).lower():
                print("建议: 模型名称可能不正确，请参考官方文档获取正确的模型名称")
                print("通义千问SDK文档: https://help.aliyun.com/zh/dashscope/developer-reference/tongyi-qianwen-2-5-7b-api-reference")
                logger.info("建议查看官方文档获取正确的模型名称")
            
            return False
            
    except Exception as e:
        print(f"❌ 检查模型可用性时出错: {str(e)}")
        logger.error(f"检查模型可用性时出错: {str(e)}")
        return False

def main():
    """主函数：生成所有对话"""
    print("\n========== 陆家嘴活动轨迹数据生成 ==========")
    print(f"开始生成 {NUM_DIALOGUES} 个对话，输出目录: {OUTPUT_DIR}")
    print(f"使用模型: {MODEL_NAME}")
    
    # 检查API密钥
    if not API_KEY or API_KEY == "your_api_key_here" or len(API_KEY) < 10:
        logger.error("API密钥未设置或无效")
        print("⚠️ API密钥未设置或无效，请检查配置")
        return
    
    # 测试API连接
    if not test_api_connection():
        retry = input("API连接测试失败。是否仍要继续？(y/n): ").strip().lower()
        if retry != 'y':
            print("程序已终止")
            return
        print("继续执行程序...")
    
    # 检查模型可用性
    if not check_api_model_availability():
        retry = input("模型可用性检查失败。是否仍要继续？(y/n): ").strip().lower()
        if retry != 'y':
            print("程序已终止")
            return
        print("继续执行程序...")
    
    # 显示已有文件数量
    existing_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('dialogue_') and f.endswith('.json')]
    logger.info(f"输出目录中已有 {len(existing_files)} 个对话文件")
    print(f"输出目录中已有 {len(existing_files)} 个对话文件")
    
    successful = 0
    index = 1
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 5
    
    # 使用进度条显示处理进度
    if TQDM_AVAILABLE:
        progress_bar = tqdm(total=NUM_DIALOGUES, desc="生成对话")
    
    start_time = time.time()
    
    try:
        while successful < NUM_DIALOGUES:
            # 检查连续失败次数
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logger.warning(f"检测到 {MAX_CONSECUTIVE_FAILURES} 次连续失败，暂停 30 秒后继续...")
                print(f"\n检测到 {MAX_CONSECUTIVE_FAILURES} 次连续失败，暂停 30 秒后继续...")
                time.sleep(30)  # 较长暂停以恢复
                consecutive_failures = 0
            
            # 尝试生成对话
            if generate_dialogue(index):
                successful += 1
                consecutive_failures = 0  # 重置连续失败计数
                if TQDM_AVAILABLE:
                    progress_bar.update(1)
                index += 1
            else:
                consecutive_failures += 1
                logger.warning(f"对话 {index} 生成失败，这是第 {consecutive_failures} 次连续失败")
                print(f"对话 {index} 生成失败，这是第 {consecutive_failures} 次连续失败")
                time.sleep(RETRY_DELAY)
            
            # 为避免请求频率限制，添加延迟
            time.sleep(RATE_LIMIT_DELAY)
            
            # 显示进度
            if not TQDM_AVAILABLE and successful > 0 and successful % 5 == 0:
                elapsed = time.time() - start_time
                rate = successful / elapsed if elapsed > 0 else 0
                estimated_total = elapsed / successful * NUM_DIALOGUES if successful > 0 else 0
                remaining = estimated_total - elapsed
                print(f"进度: {successful}/{NUM_DIALOGUES} ({successful/NUM_DIALOGUES*100:.1f}%), "
                      f"速率: {rate*60:.2f}个/分钟, 预计剩余时间: {remaining/60:.1f}分钟")
    
    except KeyboardInterrupt:
        print("\n用户中断，程序已停止")
        logger.info("用户中断，程序已停止")
    
    finally:
        if TQDM_AVAILABLE:
            progress_bar.close()
    
    # 验证生成的文件数量
    final_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('dialogue_') and f.endswith('.json')]
    logger.info(f"成功生成 {successful} 个对话，输出目录中共有 {len(final_files)} 个对话文件")
    print(f"\n成功生成 {successful} 个对话，输出目录中共有 {len(final_files)} 个对话文件")
    
    # 输出一些文件示例
    if final_files:
        sample_files = sorted(final_files)[:5]  # 取前5个文件作为示例
        logger.info(f"示例文件: {', '.join(sample_files)}")
        print(f"示例文件: {', '.join(sample_files)}")
    
    # 统计运行时间
    total_time = time.time() - start_time
    logger.info(f"总运行时间: {total_time/60:.2f} 分钟")
    print(f"总运行时间: {total_time/60:.2f} 分钟")
    
    # 如果有失败的生成，提供重试建议
    if successful < NUM_DIALOGUES:
        print("\n提示: 部分对话生成失败。您可以稍后重新运行此脚本，已成功生成的对话将被跳过。")
        logger.info("部分对话生成失败，可以稍后重新运行脚本继续生成")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断，程序已停止")
        logger.info("用户中断，程序已停止")
    except Exception as e:
        logger.critical(f"程序执行过程中发生未处理异常: {str(e)}")
        logger.critical(traceback.format_exc())
        print(f"\n程序执行过程中发生未处理异常: {str(e)}")
        print("详细错误信息已记录到日志文件")
