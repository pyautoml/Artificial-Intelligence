"""
What Are Tools?
Tools are methods (functions) provided to the model (or to an agent), making them available 
for use when the agent needs to perform actions such as collecting data from the internet, 
making calculations, retrieving emails from a mailbox, and more. Since the main goal is to 
make agents increasingly autonomous, we cannot respond to every concern an agent may have. 
 
Instead, we aim to equip the agent with tools they can use independently.
Let's explore how to build such tools and how to utilize tools that are integrated into external 
Python libraries like Ollama and OpenAI.
"""

import httpx
import calendar
import datetime
import importlib
import traceback
from typing import Callable


registered_functions = {}

def register_function(func) -> Callable:
    registered_functions[func.__name__] = func
    return func


def run_callable(name: str, arguments: dict) -> Callable | dict:
    try:
        module = importlib.import_module("tools")
        func = getattr(module, name, None)
        func = registered_functions.get(name)

        if func and callable(func):
            return func(**arguments)
        else:
            return {"error": f"Function '{name}' is not callable or not found"}
    except Exception as e:
        traceback.print_exc()
        return {"error": f"Error executing {name}: {str(e)}"}


# step 1: create a function & register it globally
@register_function
def weather(city: str) -> str:
    """
    Use this tool to check updated weather for a given city.
    Remember to replace diacritics with neutral consotants or vowels, e.g. Kraków -> Krakow You need to provide city name.

    Arguments:
    city (str): The city name.

    Returns:
    str: Response containing the weather data for the provided city, or in case of error: a str containint error message.
    """

    base_url: str = f"http://wttr.in/{city}?format=j1"
    response = httpx.Client(follow_redirects=True, default_encoding="utf-8").get(base_url)
    response.raise_for_status()

    if response.status_code == 200:
        data = response.json()
        temp = data["current_condition"][0]["temp_C"]
        return f"The temperature in {city} is {temp} Celsius degree"
    else:
        return f"Could not retrieve weather data for {city}."


@register_function
def today_is() -> str:
    """
    Use this tool to check today's time and date.
    Remember to replace diacritics with neutral consotants or vowels, e.g. Kraków -> Krakow You need to provide city name.

    Returns:
    str: A string representing timestamp made of time and date in format 'YYYY-MM-DD HH:MM:SS'.
    """
    return f"The time and date now is: {str(datetime.datetime.now().replace(microsecond=0))}"


@register_function
def day_of_week() -> str:
    """
    Use to get name of today's day.

    Arguments:
    None

    Returns: name of the day of the week for today.
    """
    d = datetime.date.today()
    weekday_index = calendar.weekday(d.year, d.month, d.day)
    weekday_names = list(calendar.day_name)
    return str(weekday_names[weekday_index].capitalize())


# step 2: add tools schema so the agent is able to use it
today_is_tool = {
    "type": "function",
    "function": {
        "name": "today_is",
        "description": "Get today's date",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
}

weather_tool = {
    "type": "function",
    "function": {
        "name": "weather",
        "description": "Get current weather for a specific city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "The city name"}},
            "required": ["city"],
        },
    },
}

day_of_week_tool = {
    "type": "function",
    "function": {
        "name": "day_of_week",
        "description": "Get today's the day of the week",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
}
