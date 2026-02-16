from typing import TypedDict

# Define dictionary schema
class Employee(TypedDict):
    name: str
    age: int

# Create instance following schema
emp: Employee = {
    "name": "Shubham",
    "age": 25
}

print(f"Name: {emp['name']}")
print(f"Age: {emp['age']}")
