from app.project2.google_llm_chat_with_memory import chat


def main():
    res = chat("Hello, my name is Ayomide. Can you tell me a joke?")
    print(f"Assistant: {res}");
    
    res = chat("Can you tell me my name")
    print(f"Assistant: {res}")
    
    print("End of Memory!")


if __name__ == "__main__":
    main()
