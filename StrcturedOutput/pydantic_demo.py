from pydantic import BaseModel, Field, EmailStr
from typing import Optional

# 1. Define Model
class User(BaseModel):
    name: str
    age: int = Field(gt=0, lt=120, description="User age between 1-119")
    email: EmailStr
    is_active: bool = True
    bio: Optional[str] = None

# 2. Coerce & Validation
# Notice "25" (string) becomes 25 (int)
user_data = {"name": "Shubham", "age": "25", "email": "test@mail.com"}
user = User(**user_data)

# 3. Pydantic Object
print(f"Object: {user}")

# 4. Convert to Dict & JSON
print(f"Dict: {user.model_dump()}")
print(f"JSON: {user.model_dump_json()}")
