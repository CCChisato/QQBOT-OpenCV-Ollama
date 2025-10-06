from service import ollama_query, ollama_query_stream, describe_image_with_ollama
from static import MODEL_NAME_CHAT, MODEL_NAME_VL, PROMPT_ROLE_CHAT, ICON1_PATH, ICON2_PATH
from context_manager import PayloadBuilder
from utils import parse_response, format_response_to_string, ScreenshotNotChangedException, update_screenshot_cache, get_new_responses
from task import describe_screen_capture, handle_response
import time
import logging

if __name__ == "__main__":
    builder = PayloadBuilder(MODEL_NAME_CHAT, stream=False)
    builder.system_prompt = PROMPT_ROLE_CHAT
    time.sleep(5)
    responce = ""
    responce_cache = ""
    while True:
        try:
            if responce:
                responce_cache = responce# 复制当前响应缓存
            responce = describe_screen_capture()
            logging.debug(f"描述屏幕截图的缓存: {responce_cache}")
            logging.debug(f"描述屏幕截图的响应: {responce}")

            result_responce = get_new_responses(responce, responce_cache)

            logging.debug(f"新响应: {result_responce}")
            formatted_response = format_response_to_string(result_responce)
            print(f"Formatted response: {formatted_response}")
            import gc
            gc.collect()
            time.sleep(1)
        except ScreenshotNotChangedException as snce:
            # 处理截图未改变的异常
            logging.error(f"截图内容未发生显著变化（可能为截图误差）: {snce}")
            time.sleep(5)  # 休眠5秒
            continue
        except Exception as e:
            # 处理其他所有异常
            logging.error(f"描述屏幕截图时出错: {e}")
            time.sleep(5)
            continue
        builder.add_user_message(formatted_response)
        logging.debug("检查builder结构："+ str(builder.build()))
        http_response = ollama_query(builder.build())
        thinking, response = parse_response(http_response)
        handle_response(response)
        update_screenshot_cache(ICON1_PATH, ICON2_PATH)
        builder.auto_summarize_and_clear()
        time.sleep(1)




