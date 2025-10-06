from typing import List, Dict, Optional, Any
from utils import image_to_base64, image_to_base64_NumPy
from service import ollama_query
from static import SUMMARIZE_PROMPT


class Message:
    def __init__(self, role: str, content: str, images: Optional[List[str]] = None):
        self.role = role
        self.content = content
        self.images: Optional[List[str]] = images  # 支持多张图

    def to_dict(self) -> Dict[str, Any]:
        message_dict = {
            "role": self.role,
            "content": self.content
        }
        if self.images is not None:
            message_dict["images"] = self.images # type: ignore
        return message_dict


class MessageManager:
    def __init__(self):
        self.messages: List[Message] = []

    def add_user_message(self, content: str) -> None:
        self.messages.append(Message("user", content))

    def add_user_message_with_imagepath(self, content: str, image_path: str) -> None:
        image_b64 = image_to_base64(image_path)
        self.messages.append(Message("user", content, [image_b64]))  # 这里是 list
    
    def add_user_message_with_image(self, content: str, image) -> None:
        image_b64 = image_to_base64(image)
        # 假设 image 是一个图片的对象
        self.messages.append(Message("user", content, [image_b64]))

    def add_user_message_with_base64(self, content: str, image_b64: str) -> None:
        self.messages.append(Message("user", content, [image_b64]))

    def add_assistant_message(self, content: str) -> None:
        self.messages.append(Message("assistant", content))

    def add_system_prompt(self, content: str) -> None:
        self.messages.append(Message("system", content))

    def clear_messages(self) -> None:
        self.messages.clear()

    def get_messages(self) -> List[Dict[str, Any]]:
        return [msg.to_dict() for msg in self.messages]


class PayloadBuilder:
    def __init__(self, model_name: str, stream: bool = False, system_prompt: Optional[str] = None, settings: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.stream = stream
        self.message_manager = MessageManager()

        self.system_prompt = system_prompt
        self.memory = ""
        self.settings = settings if settings is not None else {}

    def set_stream(self, stream: bool) -> "PayloadBuilder":
        self.stream = stream
        return self

    def add_user_message(self, content: str) -> "PayloadBuilder":
        self.message_manager.add_user_message(content)
        return self

    def add_user_message_with_image_path(self, content: str, image_path: str) -> "PayloadBuilder":
        self.message_manager.add_user_message_with_imagepath(content, image_path)
        return self
    
    def add_user_message_with_image_b64(self, content: str, image_b64: str) -> "PayloadBuilder":
        self.message_manager.add_user_message_with_base64(content, image_b64)
        return self

    def add_assistant_message(self, content: str) -> "PayloadBuilder":
        self.message_manager.add_assistant_message(content)
        return self
    
    def add_system_prompt(self, content: str) -> "PayloadBuilder":
        self.message_manager.add_system_prompt(content)
        return self

    def build(self) -> Dict[str, Any]:
        # 构建最终的 messages 列表
        built_messages = []

        # 合并 system_prompt 和 memory
        system_content = ""
        if self.system_prompt:
            system_content += self.system_prompt
        if self.memory:
            if system_content:
                system_content += "\n\n" + self.memory
            else:
                system_content = self.memory

        if system_content.strip():
            # 插入 system 消息到最前面
            built_messages.append(Message("system", system_content).to_dict())

        # 添加其余消息
        built_messages.extend(self.message_manager.get_messages())

        request =  {
            "model": self.model_name,
            "messages": built_messages,
            "stream": self.stream
        }
    
        # 应用 settings 中的配置项
        for key, value in self.settings.items():
            request[key] = value

        return request

    def reset_messages(self) -> None:
        self.message_manager.clear_messages()

    def auto_summarize_and_clear(self) -> None:
        """
        使用当前对话历史生成摘要，并重置消息历史。
        """

        # 0. 如果没有足够的消息，则不进行摘要
        MAX_TOKENS = 2048  # 假设的最大 token 数
        all_messages = self.message_manager.get_messages()
        total_length = sum(len(msg["content"]) for msg in all_messages)
        if total_length < MAX_TOKENS:
            return
        
        # 1. 复制当前 builder（不共享引用）
        summarize_builder = PayloadBuilder(
            model_name=self.model_name,
            stream=False,
            system_prompt=SUMMARIZE_PROMPT  # 专门用于自动总结的系统提示
        )

        # 2. 将当前消息复制给新 builder（除了 system_prompt）
        summarize_builder.memory = self.message_manager.memory

        for msg in self.message_manager.messages:
            if msg.role == "user":
                if msg.images:
                    # 如果有图片 Base64，需要处理成原始字符串格式
                    image_b64 = msg.images[0] if isinstance(msg.images[0], str) else ""
                    summarize_builder.add_user_message_with_base64(msg.content, image_b64)
                else:
                    summarize_builder.add_user_message(msg.content)
            elif msg.role == "assistant":
                summarize_builder.add_assistant_message(msg.content)

        # 3. 调用 Ollama 获取摘要
        try:
            summary_response = ollama_query(summarize_builder.build())
        except Exception as e:
            print(f"摘要生成失败: {e}")
            return

        # 4. 保存摘要到 memory 并清空消息历史
        self.memory = summary_response
        self.reset_messages()

    def settings_fix_loop(self):
        """
        调整设置以避免模型在处理聊天记录并总结成json格式数据的任务中进入循环。
        """
        # 假设这些是适合避免循环的默认参数
        safe_settings = {
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.9,
        "min_p": 0.05,
        "max_tokens": 300,   # 增加最大输出长度
        "repeat_penalty": 1.2,
        "tfs_z": 1.0,
        "typical_p": 1.0,
        "early_stopping": True,
        "keep_alive": "10m",  # 保持模型在内存中的时间
        }

        # 更新当前设置，保留用户自定义设置
        for key, value in safe_settings.items():
            if key not in self.settings:
                self.settings[key] = value

