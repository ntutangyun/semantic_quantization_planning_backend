import asyncio
import json
import time
from typing import Type

import websockets
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class UserChatToolInput(BaseModel):
    """Input schema for UserChatTool."""
    message: str = Field(..., description="A message or question to send to user")


class UserChatTool(BaseTool):
    name: str = "user_chat_tool"
    description: str = ("Sends a message or question to user and returns with user's response. "
                        "The input should be a message or question as a string.")
    args_schema: Type[BaseModel] = UserChatToolInput
    websocket: Type[websockets.ServerConnection] = None
    chat_status: dict = {}
    max_wait_seconds: int = 300

    def __init__(self, websocket):
        super().__init__()
        self.websocket = websocket

    def _run(self, message: str) -> str:
        """
        Sends a message or question to user and returns with user's response
        Args:
            message: The message or question to send to user
        Returns:
            User's response text
        """

        async def async_send(ws, msg):
            print("Sending message to user: " + msg)
            await ws.send(msg)
            print("Message sent to user")

        asyncio.run(async_send(self.websocket, json.dumps({
            "type": "user-chat-request",
            "data": {
                "role": "assistant",
                "content": message
            }
        })))

        self.chat_status.update({
            "pending_question": message,
            "user_response": None,
            "response_received": False
        })

        # Wait for user response with timeout
        start_time = time.time()
        while not self.chat_status["response_received"]:
            if time.time() - start_time > self.max_wait_seconds:
                raise TimeoutError("No user response received within timeout period")
            time.sleep(0.1)  # Reduce CPU usage

        return self.chat_status["user_response"]

    def receive_response(self, pending_question: str, response: str):
        """Callback for receiving user responses"""
        if self.chat_status["pending_question"] != pending_question:
            raise ValueError("Expecting response to question: " + self.chat_status[
                "pending_question"] + " but received response to question: " + pending_question)

        self.chat_status.update({
            "user_response": response,
            "response_received": True
        })
