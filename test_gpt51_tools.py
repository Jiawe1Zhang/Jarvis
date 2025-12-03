import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# 旧版 functions 格式
functions = [
  {
    "name": "get_current_weather",
    "description": "Get the current weather in a given location",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "type": "string",
          "description": "The city and state, e.g. San Francisco, CA",
        },
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
      },
      "required": ["location"],
    },
  }
]

print(f"Testing gpt-5.1 with functions (旧版参数)...")

try:
    response = client.chat.completions.create(
        model="gpt-5.1",
        messages=[{"role": "user", "content": "What's the weather like in Boston?"}],
        functions=functions,
        function_call={"name": "get_current_weather"},  # 强制调用
    )
    print("Success!")
    msg = response.choices[0].message
    print(f"Content: {msg.content}")
    print(f"Function Call: {msg.function_call}")
except Exception as e:
    print(f"Failed: {e}")
