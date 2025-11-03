from datetime import datetime

def calculate_age(birth_date: str, visit_date: str) -> int:
    birth = datetime.strptime(birth_date, "%Y-%m")
    visit = datetime.strptime(visit_date, "%Y-%m")
    age = visit.year - birth.year
    if (visit.month, visit.day if hasattr(visit, "day") else 1) < (birth.month, 1):
        age -= 1
    return age