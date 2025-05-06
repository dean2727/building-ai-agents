
from google.adk.agents import Agent
import serpapi
import os

from dotenv import load_dotenv

""" 
Put the following content into a .env file in part-1/:

SERP_API_KEY=<your DeepSeek key>
OPENAI_API_KEY=<your openAI key>
"""
load_dotenv()


def web_search(query: str) -> str:
    """Performs a real-time Google search for up-to-date information on a given query."""
    try:
        client = serpapi.Client(api_key=os.environ['SERP_API_KEY'])
        results = client.search({
            "q": query,
            "engine": "google",
            "location": "Austin,Texas",
            "hl": "en",
            "api_ke": os.environ['SERP_API_KEY']
        })
        
        # Priority 1: Answer Box
        if "answer_box" in results and "answer" in results["answer_box"]:
            return {"status": "success", "result": results["answer_box"]["answer"]}

        # Priority 2: Related Questions
        if "related_questions" in results and len(results["related_questions"]) > 0:
            best_answer = results["related_questions"][0]["snippet"]
            return {"status": "success", "result": best_answer}

        # Priority 3: Inline Videos
        if "inline_videos" in results and len(results["inline_videos"]) > 0:
            video_desc = f"{results['inline_videos'][0]['title']}: {results['inline_videos'][0]['link']}"
            return {"status": "success", "result": video_desc}

        return {"status": "error", "error_message": "No relevant results found"}            
    except Exception as e:
        return {"status": "error", "error_message": f"Search failed: {str(e)}"}


def calculator(expression: str) -> dict:
    """
    Evaluates a basic arithmetic expression and returns the result.

    Args:
        expression (str): The arithmetic expression to evaluate (e.g., "15 + 20 * 0.85").

    Returns:
        dict: A dictionary containing the calculation outcome.
              Includes a 'status' key ('success' or 'error').
              If 'success', includes a 'result' key with the computed value.
              If 'error', includes an 'error_message' key describing the issue.
    """
    try:
        # Only allow safe characters for basic arithmetic
        allowed = "0123456789+-*/.() "
        if not all(char in allowed for char in expression):
            return {"status": "error", "error_message": "Invalid characters in expression."}
        result = eval(expression, {"__builtins__": {}})
        return {"status": "success", "result": result}
    except ZeroDivisionError:
        return {"status": "error", "error_message": "Division by zero is not allowed."}
    except Exception as e:
        return {"status": "error", "error_message": f"Invalid expression: {e}"}

from google.adk.models.lite_llm import LiteLlm


ecom_agent = Agent(
    name="ecom_support_agent",
    model=LiteLlm(model="openai/gpt-4o"),
    description="Provides eCommerce answers requiring web searching or nnumber calculations.",
    instruction="""Handle customer queries for Dell E-Commerce:
    1. Math: Use calculator (e.g., ".15 * 200")
    2. Policies/Regions: Use web search (e.g., CA return policy)
    3. NEVER combine tools - pick one based on query type""",
    tools=[web_search, calculator],
)


from google.adk.sessions import InMemorySessionService
# InMemorySessionService is simple, non-persistent storage
session_service = InMemorySessionService()

# Define constants for identifying the interaction context
APP_NAME = "ecomm_app"
USER_ID = "user_1"
SESSION_ID = "session_001" # Using a fixed ID for simplicity

# Create the specific session where the conversation will happen
session = session_service.create_session(
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id=SESSION_ID
)
print(f"Session created: App='{APP_NAME}', User='{USER_ID}', Session='{SESSION_ID}'")

# --- Runner ---
from google.adk.runners import Runner
# Key Concept: Runner orchestrates the agent execution loop.
runner = Runner(
    agent=ecom_agent, # The agent we want to run
    app_name=APP_NAME,   # Associates runs with our app
    session_service=session_service # Uses our session manager
)


from google.genai import types

async def call_agent_async(query: str, runner, user_id, session_id):
  """Sends a query to the agent and prints the final response."""
  print(f"\n>>> User Query: {query}")

  # Prepare the user's message in ADK format
  content = types.Content(role='user', parts=[types.Part(text=query)])

  final_response_text = "Agent did not produce a final response." # Default

  # Key Concept: run_async executes the agent logic and yields Events.
  # We iterate through events to find the final answer.
  async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
      # You can uncomment the line below to see *all* events during execution
      print(f"  [Event] Author: {event.author}, Type: {type(event).__name__}, Final: {event.is_final_response()}, Content: {event.content}")
      
      #print("DEBUG:", (event.content if event.content else ''), "~~~~", (event.actions if event.actions else ''))

      # Key Concept: is_final_response() marks the concluding message for the turn.
      if event.is_final_response():
          if event.content and event.content.parts:
             # Assuming text response in the first part
             final_response_text = event.content.parts[0].text
          elif event.actions and event.actions.escalate: # Handle potential errors/escalations
             final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
          # Add more checks here if needed (e.g., specific error codes)
          break # Stop processing events once the final response is found

  print(f"<<< Agent Response: {final_response_text}")


async def run_tests():
    await call_agent_async("What’s the total cost of my cart (45.99) with a 15 percent discount?",
        runner=runner,
        user_id=USER_ID,
        session_id=SESSION_ID)


    await call_agent_async("What’s the return policy for electronics at Dell?",
        runner=runner,
        user_id=USER_ID,
        session_id=SESSION_ID) # Expecting the tool's error message



import asyncio
if __name__ == "__main__": # Ensures this runs only when script is executed directly
    print("Executing using 'asyncio.run()' (for standard Python scripts)...")
    try:
        # This creates an event loop, runs your async function, and closes the loop.
        asyncio.run(run_tests())
    except Exception as e:
        print(f"An error occurred: {e}")