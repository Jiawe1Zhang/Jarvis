import unittest
import os
from src.chat_openai import ChatOpenAI

class ChatOpenAITest(unittest.TestCase):
    def test_real_ollama_chat(self):
        """
        Integration test with local Ollama instance.
        Requires: 
        1. Ollama running (ollama serve)
        2. llama3.1 model pulled (ollama pull llama3.1)
        """
        print("\n========================================")
        print("Connecting to local Ollama (llama3.1)...")
        
        try:
            # Initialize with local Ollama settings
            llm = ChatOpenAI(
                model="llama3.1",
                system_prompt="You are a helpful assistant.",
                base_url="http://localhost:11434/v1",
                api_key="ollama" # Ollama doesn't check key, but library needs non-empty string
            )
            
            user_msg = "Hello! Please introduce yourself in one sentence."
            print(f"User: {user_msg}")
            
            # Send request
            result = llm.chat(user_msg)
            content = result["content"]
            
            print(f"Assistant: {content}")
            print("========================================")
            
            # Assertions
            self.assertIsNotNone(content)
            self.assertNotEqual(content, "")
            self.assertGreater(len(content), 5)
            
        except Exception as e:
            print(f"\nTest Failed: {e}")
            print("Ensure Ollama is running on port 11434 and llama3.1 is installed.")
            # We fail the test if connection fails, so you know something is wrong
            self.fail(f"Connection failed: {e}")

if __name__ == "__main__":
    unittest.main()
