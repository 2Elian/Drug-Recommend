from datetime import datetime



def calculate_age(birth_date: str, visit_date: str) -> int:
    birth = datetime.strptime(birth_date, "%Y-%m")
    visit = datetime.strptime(visit_date, "%Y-%m")
    age = visit.year - birth.year
    if (visit.month, visit.day if hasattr(visit, "day") else 1) < (birth.month, 1):
        age -= 1
    return age

def get_bmi_description(bmi_value: float, default: str) -> str:
    """根据 BMI 值返回描述（使用中国标准）"""
    if bmi_value is None:
        return default
    try:
        if bmi_value >= 28:
            return "肥胖"
        elif bmi_value >= 24:
            return "超重"
        elif bmi_value >= 18.5:
            return "正常"
        else:
            return "体重过轻"
    except (TypeError, ValueError):
        return default


