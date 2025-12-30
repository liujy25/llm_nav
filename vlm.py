import logging
import os
import torch
import numpy as np
import cv2

from PIL import Image
from utils import append_mime_tag, encode_image_b64, resize_image_if_needed


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

    def __init__(self, model="gpt-4o-latest", system_instruction=None, max_image_res=None):
        """
        Initialize the OpenAI model with specified configuration.

        Parameters
        ----------
        model : str
            The model version to be used.
        system_instruction : str, optional
            System instructions for model behavior.
        """
        from openai import OpenAI
        self.name = model
        self.client = OpenAI(
            base_url=os.environ.get("OPENAI_BASE_URL"),
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        self.model = model
        self.history = [] 
        self.max_image_res = max_image_res


    def call_chat(self, history: int, images: list[np.array], text_prompt: str):
        """
        Perform context-aware inference with the OpenAI model.

        Parameters
        ----------
        history : int
            The number of environment steps to keep in context.
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
                messages=self.history + messages,
                max_tokens=2048,
                temperature=0.0,
            )
            self.history.append(messages[0]) # append user message
            self.history.append({"role": "assistant", "content": [{"type": "text", "text": response.choices[0].message.content}]}) # append response

            # Manage history length based on the number of past steps to keep
            if len(self.history) > 2 * history:
                self.history = self.history[-2 * history:]

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

    def reset(self):
        """
        Reset the chat history.
        """
        self.history = []


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

