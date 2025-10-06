# Agent
Agent是一个基于Ollama API的智能助手，能够自动捕获屏幕内容并使用大型语言模型进行分析和交互。

<img width="2312" height="1325" alt="Screenshot 2025-10-06 113421" src="https://github.com/user-attachments/assets/54423535-7125-4dce-ae77-92a8e5d4abc1" />

## 功能特点
- 屏幕内容识别 ：自动捕获指定区域的屏幕内容并进行分析
- 聊天记录提取 ：能够从图像中识别聊天记录并转换为结构化JSON格式
- 智能交互 ：基于识别的内容提供智能回复
- 多模态支持 ：使用千问2.5vl:7b视觉语言模型进行图像理解
- 自动记忆管理 ：支持对话历史自动摘要和清理
## 技术栈
- Ollama API ：本地大语言模型服务
- Python ：核心开发语言
- OpenCV ：图像处理和分析
- PyAutoGUI ：屏幕截图和自动化操作
## 安装要求
### 系统要求
- Python 3.8+
- NVIDIA GPU (推荐用于更好的性能)
- Ollama服务 (本地运行)
### 依赖库
```
pip install requests opencv-python 
pyautogui numpy
```
### 模型要求
- 需要在Ollama中安装以下模型:
  - qwen3:14b (聊天模型)
  - qwen2.5vl:7b (视觉语言模型)
## 使用方法
1. 1.
   确保Ollama服务已启动并加载所需模型
2. 2.
   配置 static.py 中的图标路径和提示词
3. 3.
   运行主程序:
```
python agent.py
```
## 项目结构
- agent.py : 主程序入口，管理整体工作流程
- task.py : 定义图像识别和处理任务
- service.py : 封装Ollama API调用
- context_manager.py : 管理对话上下文和消息历史
- utils.py : 工具函数集合
- static.py : 静态配置和提示词模板
## 高级配置
可以通过修改 static.py 中的以下参数自定义Agent行为:

- PROMPT_CHAT_HISTORY : 图像识别提示词
- PROMPT_ROLE_CHAT : 角色扮演提示词
- MODEL_NAME_CHAT : 聊天模型名称
- MODEL_NAME_VL : 视觉语言模型名称
## 故障排除
- GPU占用过高 : 尝试在 context_manager.py 中调整 settings_fix_loop 方法的参数
- 请求超时 : 在 service.py 中增加timeout参数值
- 内存问题 : 程序已集成GC回收机制，如仍有问题可调整Ollama服务参数
## 许可证
MIT License

## 贡献指南
欢迎提交问题和拉取请求，共同改进这个项目!
