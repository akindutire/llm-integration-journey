from enum import Enum

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()

MODEL = "gemini-3.8-flash"


# ---------- Structured output schema ----------

class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Ingredient(BaseModel):
    name: str = Field(description="Ingredient name, e.g. 'long-grain parboiled rice'")
    quantity: float | None = Field(None, description="Numeric amount; null if 'to taste'")
    unit: str | None = Field(None, description="Unit of measure, e.g. 'cups', 'g', 'tbsp'")
    notes: str | None = Field(None, description="Prep notes, e.g. 'finely chopped'")


class Step(BaseModel):
    step_number: int
    instruction: str
    duration_minutes: int | None = Field(None, description="Approximate time for this step")


class Substitution(BaseModel):
    original: str = Field(description="Ingredient being replaced")
    replacement: str = Field(description="What to use instead")
    reason: str = Field(description="Which dietary restriction this satisfies, or why")


class Nutrition(BaseModel):
    calories: int | None = None
    protein_g: float | None = None
    carbs_g: float | None = None
    fat_g: float | None = None


class Recipe(BaseModel):
    dish_name: str
    description: str = Field(description="One or two sentence summary of the dish")
    cuisine: str
    servings: int
    prep_time_minutes: int
    cook_time_minutes: int
    difficulty: Difficulty
    ingredients: list[Ingredient]
    steps: list[Step]
    substitutions: list[Substitution] = Field(
        default_factory=list,
        description="Swaps made to satisfy the dietary restrictions",
    )
    dietary_compliance: list[str] = Field(
        default_factory=list,
        description="Dietary labels this recipe satisfies, e.g. 'vegan', 'gluten-free'",
    )
    nutrition_per_serving: Nutrition | None = None
    tips: list[str] = Field(default_factory=list)


# ---------- LLM call ----------

def get_dish_recipe(
    dish: str,
    dietary_restrictions: list[str] | None = None,
    servings: int | None = None,
    model: str = MODEL,
) -> Recipe:
    """Ask Gemini for a recipe and return it as a validated `Recipe` object."""
    client = genai.Client()
    restrictions = ", ".join(dietary_restrictions) if dietary_restrictions else "None"
    prompt = f"""
        You are an expert chef.
        Produce a complete, authentic recipe for the dish: {dish}.
        Dietary restrictions: {restrictions}.
        {f"Scale the recipe for {servings} servings." if servings else ""}
        If restrictions apply, adapt the recipe so it fully complies and list every
        substitution you made. Steps must be in order and numbered from 1.
    """

    print(f"Prompting Gemini with:\n{prompt}\n")
    
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Recipe,
        ),
    )

    # The SDK parses JSON into the Pydantic model; fall back to manual validation.
    if isinstance(response.parsed, Recipe):
        return response.parsed
    return Recipe.model_validate_json(response.text)
