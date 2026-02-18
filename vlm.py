import logging
import numpy as np

from PIL import Image
from utils import append_mime_tag, encode_image_b64, resize_image_if_needed
from openai import OpenAI

class VLM:
    """
    Base class for a Vision-Language Model (VLM) agent.
    This class should be extended to implement specific VLMs.
    """

    def __init__(self, **kwargs):
        """
        Initializes the VLM agent with optional parameters.
        """
        self.name = "not implemented"

    def call(self, images: list[np.array], text_prompt: str):
        """
        Perform inference with the VLM agent, passing images and a text prompt.

        Parameters
        ----------
        images : list[np.array]
            A list of RGB image arrays.
        text_prompt : str
            The text prompt to be processed by the agent.
        """
        raise NotImplementedError

    def call_chat(self, history: int, images: list[np.array], text_prompt: str):
        """
        Perform context-aware inference with the VLM, incorporating past context.

        Parameters
        ----------
        history : int
            The number of context steps to keep for inference.
        images : list[np.array]
            A list of RGB image arrays.
        text_prompt : str
            The text prompt to be processed by the agent.
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset the context state of the VLM agent.
        """
        pass

    def rewind(self):
        """
        Rewind the VLM agent one step by removing the last inference context.
        """
        pass

    def get_spend(self):
        """
        Retrieve the total cost or spend associated with the agent.
        """
        return 0


class OpenAIVLM(VLM):
    """
    An implementation using models served via OpenAI API.
    """

    def __init__(self, model="gpt-4o-latest", system_instruction=None, max_image_res=None,
                 api_key="EMPTY", base_url="http://10.15.89.71:34134/v1/", timeout=10):
        """
        Initialize the OpenAI model with specified configuration.

        Parameters
        ----------
        model : str
            The model version to be used.
        system_instruction : str, optional
            System instructions for model behavior.
        max_image_res : int, optional
            Maximum image resolution for resizing.
        api_key : str, optional
            API key for OpenAI client (default: "EMPTY" for local deployment).
        base_url : str, optional
            Base URL for OpenAI API (default: local vLLM server).
        timeout : int, optional
            Request timeout in seconds (default: 10).
        """
        self.name = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout
        )
        self.model = model
        self.system_instruction = system_instruction  # Store system instruction
        self.initial_prompt = None  # Will be set during reset with goal info
        self.history = []  # Conversation history (user+assistant pairs)
        self.max_image_res = max_image_res


    def call_chat(self, history: int, images: list[np.array], text_prompt: str):
        """
        Perform context-aware inference with the OpenAI model.
        
        History structure:
        1. System instruction (if provided, always at the beginning)
        2. Initial prompt (if provided via reset(), contains full task briefing)
        3. Recent conversation history (user+assistant pairs, limited by history parameter)
        4. Current message

        Parameters
        ----------
        history : int
            The number of recent conversation ROUNDS (user+assistant pairs) to keep in context.
            Note: System instruction and initial prompt are always kept.
        images : list[np.array]
            Current observation images (typically just one image per iteration).
        text_prompt : str
            The text prompt for current iteration (typically short).
        """
        # Build current user message
        text_contents = [{
            "type": "text",
            "text": text_prompt
        }]
        image_contents = self._image_contents_from_images(images)
        current_message = {"role": "user", "content": text_contents + image_contents}
        
        # Construct full message list for API call
        messages = []
        
        # 1. Add system instruction (if available)
        if self.system_instruction:
            messages.append({"role": "system", "content": self.system_instruction})
        
        # 2. Add initial prompt as first user message (if available)
        if self.initial_prompt:
            messages.append({"role": "user", "content": self.initial_prompt})
            # Add a placeholder assistant acknowledgment if this is the first iteration
            if len(self.history) == 0:
                messages.append({"role": "assistant", "content": "Understood. I will follow these instructions for navigation."})
        
        # 3. Add recent conversation history (limited by history parameter)
        # Only keep the most recent N rounds of conversation
        if len(self.history) > 2 * history:
            self.history = self.history[-2 * history:]
        messages.extend(self.history)
        
        # 4. Add current message
        messages.append(current_message)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=2048,
                temperature=0.0,
            )
            
            # Save current interaction to history
            self.history.append(current_message)
            self.history.append({
                "role": "assistant", 
                "content": [{"type": "text", "text": response.choices[0].message.content}]
            })

        except Exception as e:
            logging.error(f"OPENAI API ERROR: {e}")
            return "OPENAI API ERROR"

        return response.choices[0].message.content

    def call_chat_with_custom_history(self, custom_history: list, images: list[np.array], text_prompt: str):
        """
        使用自定义历史调用VLM（由NavAgent控制Memory）
        
        这个方法允许NavAgent完全控制传递给VLM的对话历史，而不是使用self.history。
        用于实现关键帧/非关键帧的分层Memory机制。
        
        Parameters
        ----------
        custom_history : list
            自定义的对话历史，格式为：
            [
                {"role": "user", "content": [...]},
                {"role": "assistant", "content": "..."},
                ...
            ]
        images : list[np.array]
            当前观测图片
        text_prompt : str
            当前prompt
            
        Returns
        -------
        str
            VLM响应文本
        """
        # 构建当前用户消息
        text_contents = [{
            "type": "text",
            "text": text_prompt
        }]
        image_contents = self._image_contents_from_images(images)
        current_message = {"role": "user", "content": text_contents + image_contents}
        
        # 构建完整消息列表
        messages = []
        
        # 1. 添加系统指令（如果有）
        if self.system_instruction:
            messages.append({"role": "system", "content": self.system_instruction})
        
        # 2. 添加初始prompt（如果有）
        if self.initial_prompt:
            messages.append({"role": "user", "content": self.initial_prompt})
            messages.append({"role": "assistant", "content": "Understood. I will follow these instructions for navigation."})
        
        # 3. 添加自定义历史（关键：由NavAgent控制）
        messages.extend(custom_history)
        
        # 4. 添加当前消息
        messages.append(current_message)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=2048,
                temperature=0.0,
            )
            
            # 注意：不保存到self.history，因为Memory由NavAgent管理
            
        except Exception as e:
            logging.error(f"OPENAI API ERROR: {e}")
            return "OPENAI API ERROR"

        return response.choices[0].message.content


    def rewind(self):
        """
        Rewind the chat history by one step.
        """
        if len(self.history) > 1:
            self.history = self.history[:-2]

    def reset(self, initial_prompt: str = None):
        """
        Reset the chat history and optionally set initial prompt.
        
        Parameters
        ----------
        initial_prompt : str, optional
            Initial prompt containing task briefing (e.g., goal description, rules).
            This will be added to history after system instruction.
        """
        self.history = []
        self.initial_prompt = initial_prompt


    def call(self, images: list[np.array], text_prompt: str):
        """
        Perform contextless inference with the Gemini model.

        Parameters
        ----------
        images : list[np.array]
            A list of RGB image arrays.
        text_prompt : str
            The text prompt to process.
        """
        text_contents = [{
            "type": "text",
            "text": text_prompt
        }]
        image_contents = self._image_contents_from_images(images)
        messages = [{"role": "user", "content": text_contents + image_contents}]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )

        except Exception as e:
            logging.error(f"OPENAI API ERROR: {e}")
            return "OPENAI API ERROR"

        return response.choices[0].message.content


    def get_spend(self):
        """
        Retrieve the total spend on model usage.
        NOTE: not implemented
        """
        return 0


    def _image_contents_from_images(self, images: list[np.ndarray]):
        image_contents = [
            {
                "type": "image_url",
                "image_url":
                {
                    "url": append_mime_tag(encode_image_b64(Image.fromarray(image[:, :, :3], mode='RGB'))) \
                    if self.max_image_res is None else append_mime_tag(encode_image_b64(
                        resize_image_if_needed(Image.fromarray(image[:, :, :3], mode='RGB'), self.max_image_res)))
                }
            }
            for image in images
        ]
        return image_contents

