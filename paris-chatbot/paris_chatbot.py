import os
from openai import OpenAI

# Define the model to use
model = "gpt-4o-mini"

# Define the client
client = OpenAI()

# Define the questions
questions = [
    "How far away is the Louvre from the Eiffel Tower (in miles) if you are driving?",
    "Where is the Arc de Triomphe?",
    "What are the must-see artworks at the Louvre Museum?"
]

# Initialize the conversation with a system message
conversation = [
    {
        "role": "system",
        "content": (
            "You are a knowledgeable and enthusiastic virtual travel guide for Paris, France, "
            "working for Peterman Reality Tours. You provide accurate, concise, and engaging "
            "information about Parisian landmarks, attractions, and travel tips. "
            "Keep your responses informative yet brief."
        )
    }
]

# Loop through each question, get response, and add both to the conversation
for question in questions:
    # Format the question string as a user message
    user_message = {"role": "user", "content": question}

    # Add the question to the conversation
    conversation.append(user_message)

    # Fetch a response from the model
    response = client.chat.completions.create(
        model=model,
        messages=conversation,
        temperature=0.0,
        max_tokens=100
    )

    # Extract the text from the model's response
    assistant_reply = response.choices[0].message.content

    # Add the response to the conversation
    conversation.append({"role": "assistant", "content": assistant_reply})

# Print the full conversation
for message in conversation:
    role = message["role"].upper()
    content = message["content"]
    print(f"[{role}]\n{content}\n")