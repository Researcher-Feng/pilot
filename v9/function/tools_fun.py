"""
Tools module - simplified for raw API calls.
Tool functions are kept as plain functions without LangChain decorators.
"""
import sys
import random
sys.path.append(r'../..')
from function.contex_fun import Context


def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"


def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


def get_user_location(context: Context) -> str:
    """Retrieve user information based on user ID."""
    user_id = context.user_id
    return "Florida" if user_id == "1" else "SF"


def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


def parallel_thinking(query: str) -> str:
    """Conduct parallel thinking for complex problems.

    Use this when you need to:
    - Consider multiple approaches simultaneously
    - Compare different solution strategies
    - Analyze a problem from various perspectives
    - Evaluate alternative hypotheses
    """
    thinking_prompts = [
        "Let me consider this from different angles...",
        "Approach 1: Mathematical solution. Approach 2: Logical reasoning. Approach 3: Real-world application.",
        "I'm thinking about this problem in parallel: algebraic method, geometric method, and computational method.",
        "Considering multiple perspectives: conceptual understanding, procedural knowledge, and application skills."
    ]
    return random.choice(thinking_prompts)


def socratic_questioning(problem_statement: str) -> str:
    """Use Socratic questioning to guide thinking without giving direct answers.

    This tool helps:
    - Clarify concepts through questioning
    - Challenge assumptions
    - Explore implications
    - Examine evidence and reasons
    """
    socratic_questions = [
        "What do you already know about this type of problem?",
        "Can you break this problem down into smaller parts?",
        "What would happen if you tried a different approach?",
        "Why do you think that step is correct?",
        "What evidence supports your current thinking?",
        "Have you considered any alternative methods?",
        "How could you verify if your answer is reasonable?"
    ]
    return f"Socratic guidance: {random.choice(socratic_questions)}"


def math_concept_explainer(concept: str, level: str = "intermediate") -> str:
    """Explain mathematical concepts at different difficulty levels.

    Args:
        concept: The mathematical concept to explain
        level: beginner, intermediate, or advanced
    """
    explanations = {
        "algebra": {
            "beginner": "Algebra is like a puzzle where we use letters to represent unknown numbers.",
            "intermediate": "Algebra involves solving equations to find unknown variables using operations like addition, subtraction, multiplication, and division.",
            "advanced": "Algebra studies mathematical symbols and the rules for manipulating these symbols to solve equations and understand relationships between variables."
        },
        "geometry": {
            "beginner": "Geometry is about shapes, sizes, and positions of things.",
            "intermediate": "Geometry deals with properties and relationships of points, lines, angles, surfaces, and solids.",
            "advanced": "Geometry is the branch of mathematics concerned with the properties and relations of points, lines, surfaces, solids, and higher dimensional analogs."
        }
    }

    if concept.lower() in explanations:
        return explanations[concept.lower()].get(level, explanations[concept.lower()]["intermediate"])
    else:
        return f"Explanation for {concept} at {level} level: This concept involves mathematical principles that can be understood through practice and application."