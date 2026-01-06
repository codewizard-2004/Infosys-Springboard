from groq import Groq
import os
import sys


# Add the project root directory to sys.path to resolve the 'config' module
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from config import settings
except ImportError:
    # Fallback in case the above fails
    sys.path.insert(0, project_root)
    from config import settings

def get_food_description(food_name: str, model="llama-3.3-70b-versatile"):
    """
    Get food description using the groq api.
    """
    client = Groq(api_key=settings.GROQ_API_KEY)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": f"Give a small description, recipe details and some cooking tips for {food_name}?"}
        ]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    print(get_food_description("pizza"))

    