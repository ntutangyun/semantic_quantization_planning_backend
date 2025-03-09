import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
import re

import websockets
from crewai.tasks import TaskOutput

from user_chat_tool import UserChatTool
from user_interviewer_crew import init_user_interviewer_crew, InterviewSummaryDataModel


async def handler(websocket):
    user_chat_tool = UserChatTool(websocket)

    def user_interview_callback(task_output: TaskOutput):
        print("User interview completed: " + task_output.raw)

        markdown_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    
        # Search for the JSON content inside the Markdown block
        match = re.search(markdown_pattern, task_output.raw)
        
        if match:
            # Extract the JSON content
            json_content = match.group(1).strip()
        else:
            # If no Markdown block is found, assume the entire text is JSON
            json_content = task_output.raw.strip()

        interview_summary = InterviewSummaryDataModel.parse_raw(json_content)
        print("Interview summary: " + str(interview_summary))

        # send the final result and close the websocket
        asyncio.run(
            websocket.send(
                json.dumps(
                    {
                        "type": "user-interview-summary",
                        "data": interview_summary.model_dump(),
                    }
                )
            )
        )

    user_interviewer_crew = init_user_interviewer_crew(
        user_chat_tool, user_interview_callback
    )

    asyncio.create_task(user_interviewer_crew.kickoff_async())

    async for message in websocket:
        print(f"Received: {message}")
        message = json.loads(message)
        print("message in json:" + str(message))

        if "type" not in message:
            await websocket.send(
                json.dumps({"type": "error", "message": "No type in message"})
            )
            continue

        if message["type"] == "user-chat-response":
            # check if message.data is a valid list or not
            if (
                "data" not in message
                or message["data"] is None
                or "pending_question" not in message["data"]
                or "content" not in message["data"]
            ):
                await websocket.send(
                    {"type": "error", "message": "Invalid data in message"}
                )
                continue
            print(
                "user response for pending question: "
                + message["data"]["pending_question"]
                + " is: "
                + message["data"]["content"]
            )
            user_chat_tool.receive_response(
                pending_question=message["data"]["pending_question"],
                response=message["data"]["content"],
            )


async def main():
    async with websockets.serve(handler, "localhost", 8765):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())
