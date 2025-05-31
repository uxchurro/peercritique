#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.8"
# dependencies = ["openai", "python-dotenv"]
# ///

import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Read config from env
api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
model = os.getenv("OPENAI_API_MODEL_NAME", "gpt-4o")

if not api_key:
    raise ValueError("Please add your OpenAI API key to the .env file.")

# Set up OpenAI client
client = OpenAI(api_key=api_key, base_url=api_base)

# System prompt
system_prompt = """
You are a helpful assistant that critiques student feedback on UX design presentations. Help students make their feedback clearer, more specific, and more constructive.

Focus on areas like: goals, information structure, navigation, visual design, text clarity, accessibility, interactions, responsiveness, and readiness for implementation.

Respond in this format:
Overall Assessment:
...

Suggestions for Improvement:
- ...
- ...
"""

def critique_feedback(feedback_text: str):
    """
    Generate a critique of student feedback on a UX design.
    Args:
        feedback_text (str): Raw student feedback text
    Yields:
        str: Streamed response from OpenAI
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": feedback_text},
        ],
        stream=True,
    )

    for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

if __name__ == "__main__":
    # Example feedback (replace this with real input later)
    example_feedback = """
    I think the design looks pretty good. Maybe the colors could be brighter? Also, not sure if the menu was easy to find.
    """

    for delta in critique_feedback(example_feedback):
        print(delta, end="")