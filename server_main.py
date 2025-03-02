import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import websockets
from crewai.tasks import TaskOutput

from user_chat_tool import UserChatTool
from user_interviewer_crew import init_user_interviewer_crew


async def handler(websocket):
    user_chat_tool = UserChatTool(websocket)

    def user_interview_callback(final_answer: TaskOutput):
        print("User interview completed: " + final_answer.raw)
        # send the final result and close the websocket
        asyncio.run(websocket.send(
            json.dumps({"type": "user-interview-completed",
                        "data": {"role": "assistant",
                                 "content": final_answer.raw + "\n\n Interview Completed. Closing the connection."}})))

        # rely on the frontend to close the connection.
        # asyncio.run(websocket.close())

    user_interviewer_crew = init_user_interviewer_crew(user_chat_tool, user_interview_callback)

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:

        # Run the async function in a thread
        loop.run_in_executor(executor, asyncio.run, user_interviewer_crew.kickoff_async())

        async for message in websocket:
            print(f"Received: {message}")
            message = json.loads(message)
            print("message in json:" + str(message))

            if "type" not in message:
                await websocket.send(json.dumps({"type": "error", "message": "No type in message"}))
                continue

            if message["type"] == "user-chat-response":
                # check if message.data is a valid list or not
                if "data" not in message or message["data"] is None or "pending_question" not in message[
                    "data"] or "content" not in message["data"]:
                    await websocket.send({"type": "error", "message": "Invalid data in message"})
                    continue
                print("user response for pending question: " + message["data"]["pending_question"] + " is: " +
                      message["data"]["content"])
                user_chat_tool.receive_response(pending_question=message["data"]["pending_question"],
                                                response=message["data"]["content"])

        # await future  # Wait for the long-running task to complete


async def main():
    async with websockets.serve(handler, "localhost", 8765):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())
