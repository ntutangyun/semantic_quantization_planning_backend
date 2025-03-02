from crewai import Agent, Task, Crew


def init_user_interviewer_crew(user_chat_tool, final_callback=None):
    interview_agent = Agent(
        role="User Interviewer",
        goal="Estimate the user's contribution potential to the federated learning model in a smart home hub.",
        backstory="Expert in conducting technical interviews with careful follow-up questioning. "
                  "Use the user_chat_tool to send a message.",
        tools=[user_chat_tool],
        verbose=True,
    )

    interview_task = Task(
        description="""Conduct user interview to learn about where and how the user uses his/her smart home hub.
            You want to collect as much details as possible in the following aspects:
            - smart home hub's environment (e.g., for noise level estimation)
            - frequency of user-hub interactions
            - types of user-hub interactions
            """,
        expected_output="Summary of the user's smart home hub usage and the user's contribution potential (based on the quality of the user audio samples) to the federated learning model.",
        agent=interview_agent,
        # async_execution=True  # Required for waiting on user input
        callback=final_callback,
    )

    crew = Crew(
        agents=[interview_agent],
        tasks=[interview_task],
        verbose=True
    )

    return crew
