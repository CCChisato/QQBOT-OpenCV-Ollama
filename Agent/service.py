import requests
import logging
import json

from static import OLLAMA_API, MODEL_NAME_VL
from utils import image_to_base64

# 配置日志：输出到控制台
logging.basicConfig(
    level=logging.DEBUG,  # 默认只显示 INFO 及以上，避免 DEBUG 太多干扰
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)




def ollama_query(payload :dict):
    """
    使用 Ollama API 进行查询
    :param payload: 请求体，包含模型名称、消息等信息
    """

    logging.debug(f"发送给 Ollama 的请求体: {payload}")

    # 发送 POST 请求
    response = requests.post(OLLAMA_API, json=payload, timeout=180)

    logging.debug(f"HTTP 状态码: {response.status_code}")
    
    if response.status_code == 200:
        try:
            result = response.json()
            logging.debug(f"完整响应内容: {result}")

            reply = result.get("message", {}).get("content", "")
            return reply
        except Exception as e:
            logging.error(f"解析 JSON 出错: {e}")
            return "解析 AI 返回内容时出错"
    else:
        error_text = response.text
        logging.error(f"API 请求失败: {error_text}")
        return f"请求失败，状态码：{response.status_code}, 错误信息：{error_text}"
    
def ollama_query_stream(payload :dict):
    """
    使用 Ollama API 进行流式查询
    :param payload: 请求体，包含模型名称、消息等信息
    """

    logging.debug(f"发送给 Ollama 的请求体: {payload}")
    response = requests.post(OLLAMA_API, json=payload, stream=True, timeout=180)

    logging.debug(f"HTTP 状态码: {response.status_code}")

    full_reply = ""
    if response.status_code == 200:
        for line in response.iter_lines():
            if line:
                try:
                    json_line = line.decode('utf-8')
                    logging.debug(f"收到的流式数据: {json_line}")
                    data = json.loads(json_line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        full_reply += content
                        print(content, end="", flush=True)  # 实时打印
                except Exception as e:
                    logging.warning(f"解析流式数据出错: {e}")
        return full_reply
    else:
        logging.error(f"API 请求失败: {response.text}")
        return f"请求失败，状态码：{response.status_code}"
    
def describe_image_with_ollama(payload: dict):
    """
    使用 qwen2.5vl 模型对图片进行描述（可扩展为 OCR 或其他任务）
    :param image_path: 图像文件路径
    :param prompt: 提供给 AI 的指令（默认是“请描述这张图片的内容”）
    :return: AI 返回的回答内容
    """


    logging.debug(f"-发送给 Ollama 的图像请求-")

    #发起请求
    response = requests.post(OLLAMA_API, json=payload, timeout=180)

    logging.debug(f"HTTP 状态码: {response.status_code}")

    if response.status_code == 200:
        try:
            result = response.json()
            logging.debug(f"完整响应内容: {result}")
            reply = result.get("message", {}).get("content", "")
            return reply
        except Exception as e:
            logging.error(f"解析 JSON 出错: {e}")
            return "解析 AI 返回内容时出错"
    else:
        error_text = response.text
        logging.error(f"API 请求失败: {error_text}")
        return f"请求失败，状态码：{response.status_code}, 错误信息：{error_text}"