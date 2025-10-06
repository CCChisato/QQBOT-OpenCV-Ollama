import base64
import pyautogui
import cv2
import numpy as np
import json
import re
import os
import pyperclip
import time
from typing import Union

# 自定义异常类
class ScreenshotNotChangedException(Exception):
    pass
# 比较两张图像是否几乎相同（允许一定误差）
def is_similar(img1, img2, threshold=500):
    if img1.shape != img2.shape:
        # 尺寸不一致时默认认为不同
        return False
    
    diff = cv2.absdiff(img1, img2)
    non_zero = cv2.countNonZero(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY))
    
    return non_zero < threshold
def format_response_to_string(response: Union[str, list]) -> str:
    """
    清理 response 中的 Markdown 代码块标记，并解析 JSON，
    或者直接使用已解析的列表，最后格式化为 sender:message 的字符串。

    :param response: str（可能包含 Markdown 包裹的 JSON 字符串）或 list（消息列表）
    :return: str, 每条消息一行，格式为 "sender:message"
    """
    if isinstance(response, str):
        # 如果是字符串，尝试移除 Markdown 并解析 JSON
        cleaned = re.sub(r'^\s*```json\s*', '', response, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s*```\s*$', '', cleaned)

        try:
            data = json.loads(cleaned.strip())
        except json.JSONDecodeError as e:
            raise ValueError(f"清理后的 JSON 字符串无效: {e}")

        if not isinstance(data, list):
            raise TypeError("输入的 JSON 数据不是列表类型")

    elif isinstance(response, list):
        # 如果是列表，直接使用
        data = response
    else:
        raise ValueError("输入的响应内容必须是字符串或列表")

    result = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise TypeError(f"第 {idx} 条数据不是字典类型")
        sender = item.get('sender', '[未知发送者]')
        message = item.get('message', '[无消息内容]')
        result.append(f"{sender}:{message}")

    return '\n'.join(result)
def image_to_base64(file_path):
    """将图像文件转为 base64 编码"""
    with open(file_path, "rb") as image_file:
        encoded_str = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_str

def image_to_base64_NumPy(image: np.ndarray, format='png') -> str:
    """
    将 NumPy 图像数组转换为 Base64 编码的字符串。
    
    :param image: NumPy 数组形式的图像（如 OpenCV 加载的 BGR 图像）
    :param format: 图像格式，支持 'png' 或 'jpeg'
    :return: Base64 编码字符串
    """
    # 将图像编码为内存中的字节流
    success, encoded_image = cv2.imencode(f'.{format}', image)
    if not success:
        raise ValueError("图像编码失败，请检查图像格式或数据是否正确。")

    # 转换为 base64 字符串
    return base64.b64encode(encoded_image.tobytes()).decode('utf-8')

def parse_response(http_response : str) -> tuple[str, str]: 
    """ 
    解析模型响应，分离思考内容和答案 
    """ 
    thinking_start = http_response.find("<think>\n") #一般在最开头，是<think>\n 
    thinking_end = http_response.find("</think>\n\n") #中间的某个部分，似乎是</think>\n\n 
    
    if thinking_start == -1 or thinking_end == -1 or thinking_start >= thinking_end:
        return http_response, "" # 如果无法找到标签或顺序不对，则返回原始响应和空字符串

    thinking = http_response[thinking_start + len("<think>\n"):thinking_end]#两者中间的内容 
    response = http_response[thinking_end + len("</think>\n\n"): ]#剩下的内容 
    
    return thinking, response

def capture_between_icons(icon1_path, icon2_path, threshold=0.8, output_path="capture_result.png"):
    # 截取整个屏幕
    screenshot = pyautogui.screenshot()
    screen_img = np.array(screenshot)
    screen_img = cv2.cvtColor(screen_img, cv2.COLOR_RGB2BGR)

    # 加载图标模板
    icon1 = cv2.imread(icon1_path, cv2.IMREAD_COLOR)
    icon2 = cv2.imread(icon2_path, cv2.IMREAD_COLOR)

    if icon1 is None or icon2 is None:
        raise FileNotFoundError("无法加载图标文件，请确认路径是否正确。")

    # 获取图标尺寸
    h1, w1 = icon1.shape[:2]
    h2, w2 = icon2.shape[:2]

    # 使用模板匹配查找图标位置
    res1 = cv2.matchTemplate(screen_img, icon1, cv2.TM_CCOEFF_NORMED)
    loc1 = np.where(res1 >= threshold)

    res2 = cv2.matchTemplate(screen_img, icon2, cv2.TM_CCOEFF_NORMED)
    loc2 = np.where(res2 >= threshold)

    try:
        # icon1: 取所有匹配中最左上角的点 → 左上角
        y1_all, x1_all = loc1
        if len(x1_all) == 0:
            raise ValueError("icon1 未找到匹配项")
        min_x = np.min(x1_all)
        min_y = np.min(y1_all)
        x1_right_bottom = min_x + w1
        y1_right_bottom = min_y + h1

        # icon2: 取所有匹配点中最上面一行中最右边的点 → 右上角
        y2_all, x2_all = loc2
        if len(x2_all) == 0:
            raise ValueError("icon2 未找到匹配项")
        min_y_icon2 = np.min(y2_all)
        top_row_mask = (y2_all == min_y_icon2)
        x_rightmost = np.max(x2_all[top_row_mask])
        x2_right_top = x_rightmost + w2
        y2_right_top = min_y_icon2

        # 截图区域左上角 = icon1 的右下角
        left = x1_right_bottom
        top = y1_right_bottom
        # 截图区域右下角 = icon2 的右上角
        right = x2_right_top
        bottom = y2_right_top

    except IndexError:
        raise ValueError("未找到图标，请调整阈值或确保图标在屏幕上可见。")

    # 截取区域
    region = screen_img[top:bottom, left:right]

    # 检查是否存在缓存图像
    if os.path.exists(output_path):
        cached_img = cv2.imread(output_path)
        if cached_img is not None and is_similar(region, cached_img):
            raise ScreenshotNotChangedException("截图内容未发生显著变化（可能为截图误差）。")
    
    # 保存结果
    cv2.imwrite(output_path, region)
    print(f"截图已保存为 {output_path}")
    
    # 返回 Base64 编码
    return image_to_base64_NumPy(region)

def update_screenshot_cache(icon1_path, icon2_path, threshold=0.8, output_path="capture_result.png"):
    # 截取整个屏幕
    screenshot = pyautogui.screenshot()
    screen_img = np.array(screenshot)
    screen_img = cv2.cvtColor(screen_img, cv2.COLOR_RGB2BGR)

    # 加载图标模板
    icon1 = cv2.imread(icon1_path, cv2.IMREAD_COLOR)
    icon2 = cv2.imread(icon2_path, cv2.IMREAD_COLOR)

    if icon1 is None or icon2 is None:
        raise FileNotFoundError("无法加载图标文件，请确认路径是否正确。")

    # 获取图标尺寸
    h1, w1 = icon1.shape[:2]
    h2, w2 = icon2.shape[:2]

    # 使用模板匹配查找图标位置
    res1 = cv2.matchTemplate(screen_img, icon1, cv2.TM_CCOEFF_NORMED)
    loc1 = np.where(res1 >= threshold)

    res2 = cv2.matchTemplate(screen_img, icon2, cv2.TM_CCOEFF_NORMED)
    loc2 = np.where(res2 >= threshold)

    try:
        # icon1: 取所有匹配中最左上角的点 → 左上角
        y1_all, x1_all = loc1
        if len(x1_all) == 0:
            raise ValueError("icon1 未找到匹配项")
        min_x = np.min(x1_all)
        min_y = np.min(y1_all)
        x1_right_bottom = min_x + w1
        y1_right_bottom = min_y + h1

        # icon2: 取所有匹配点中最上面一行中最右边的点 → 右上角
        y2_all, x2_all = loc2
        if len(x2_all) == 0:
            raise ValueError("icon2 未找到匹配项")
        min_y_icon2 = np.min(y2_all)
        top_row_mask = (y2_all == min_y_icon2)
        x_rightmost = np.max(x2_all[top_row_mask])
        x2_right_top = x_rightmost + w2
        y2_right_top = min_y_icon2

        # 截图区域左上角 = icon1 的右下角
        left = x1_right_bottom
        top = y1_right_bottom
        # 截图区域右下角 = icon2 的右上角
        right = x2_right_top
        bottom = y2_right_top

    except IndexError:
        raise ValueError("未找到图标，请调整阈值或确保图标在屏幕上可见。")

    # 截取区域
    region = screen_img[top:bottom, left:right]
    
    # 保存结果
    cv2.imwrite(output_path, region)
    print(f"缓存已更新为 {output_path}")
    
def send_message(response: str) -> None:
    """
    将 response 写入剪贴板，并模拟 Ctrl+V 粘贴、Ctrl+Enter 发送。
    """
    # 写入剪贴板
    pyperclip.copy(response)

    # 稍微延迟一下，确保剪贴板准备好
    time.sleep(0.5)

    # 模拟 Ctrl + V（粘贴）
    pyautogui.hotkey('ctrl', 'v')

    # 模拟 Ctrl + Enter（发送）
    pyautogui.hotkey('ctrl', 'enter')

    print("消息已发送")
    
def get_new_responses(responce, responce_cache):
    """
    返回 responce 中独有的新消息（不在 responce_cache 中的消息）
    
    参数:
        responce (list): 当前获取到的消息列表（JSON 格式）
        responce_cache (list): 缓存中的旧消息列表（JSON 格式）

    返回:
        list: 只包含新增的消息列表
    """
    # 将缓存转换成集合，用于快速查找
    responce = parse_json_from_markdown(responce)
    responce_cache = parse_json_from_markdown(responce_cache)
    cache_set = {(item["sender"], item["message"]) for item in responce_cache if isinstance(item, dict)}

    # 遍历当前响应，筛选出缓存中没有的新消息
    new_responses = [
        item for item in responce
        if isinstance(item, dict) and (item["sender"], item["message"]) not in cache_set
    ]

    return new_responses

def parse_json_from_markdown(text, default=list):
    """
    从可能带有 Markdown 代码块包裹的字符串中提取 JSON 数据。
    如果输入为空或无效，默认返回空列表或空字典。

    参数:
        text (str): 可能包含 ```json ... ``` 包裹的字符串
        default: 默认返回类型，list -> [], dict -> {}

    返回:
        list 或 dict: 解析后的 JSON 数据，失败则返回默认值
    """
    if not isinstance(text, str) or text.strip() == "":
        return [] if default == list else {} if default == dict else None

    # 去除 Markdown 的 ```json 包裹
    json_str = re.sub(r'^\s*```json\s*', '', text, flags=re.DOTALL)
    json_str = re.sub(r'\s*```\s*$', '', json_str, flags=re.DOTALL)

    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        # 可选：记录日志
        # logging.warning(f"JSON 解析失败: {e}")
        return [] if default == list else {} if default == dict else None

if __name__ == "__main__":
    from static import ICON1_PATH, ICON2_PATH
    capture_between_icons(ICON1_PATH, ICON2_PATH)