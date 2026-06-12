from ollama import chat

print("Sending request...")

response = chat(
    model="qwen3:8b",
    messages=[
        {
            "role": "user",
            "content": "Hello"
        }
    ]
)

print("Response received!")
print(response["message"]["content"])