from service import ollama_query, ollama_query_stream, describe_image_with_ollama
from static import MODEL_NAME_CHAT, MODEL_NAME_VL, ICON1_PATH, ICON2_PATH, PROMPT_CHAT_HISTORY,PROMPT_CHAT_READ
from utils import image_to_base64, parse_response, capture_between_icons, send_message, ScreenshotNotChangedException
from context_manager import PayloadBuilder
import logging
import sys
import time



def describe_screen_capture(prompt = PROMPT_CHAT_READ) -> str:
    """
    使用 Ollama API 描述图像内容
    :param image_path: 图像文件路径
    :return: 图像描述
    """
    payload = PayloadBuilder(MODEL_NAME_VL, stream=False)
    payload.settings_fix_loop()
    try:
        screen_capture = capture_between_icons(ICON1_PATH, ICON2_PATH)
        payload.add_user_message_with_image_b64(prompt, screen_capture)
        logging.debug(f"-发送给 Ollama 的图像请求describe_screen_capture()-")

        # 添加重试机制
        max_retries = 3
        retry_count = 0
        response = ""
        while retry_count < max_retries:
            try:
                response = describe_image_with_ollama(payload.build())
                break  # 如果请求成功，跳出循环
            except TimeoutError as e:
                retry_count += 1
                logging.warning(f"Ollama API 请求超时 (尝试 {retry_count}/{max_retries}): {e}")
                if retry_count >= max_retries:
                    logging.error(f"Ollama API 请求多次超时: {e}")
                    response = "请求超时，请稍后重试"
                time.sleep(2)  # 等待一段时间再重试

        return response
    except ScreenshotNotChangedException as snce:
        raise snce
    except Exception as e:
        logging.error(f"描述屏幕截图时出错: {e}")
        return "描述屏幕截图时出错，请检查日志获取更多信息"
    
    
def handle_response(response: str) -> None:
    """
    处理模型的指令
    """
    if "[quit]" in response:
        print("对话结束")
        sys.exit(0)
        return
    
    if "[reject]" in response:
        print("拒绝执行该指令")
        return
    send_message(response)

    
