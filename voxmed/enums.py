from enum import Enum


class StringEnum(str, Enum):
    def __str__(self) -> str:
        return str(self.value)


class TriageLevel(StringEnum):
    ROUTINE = "Routine"
    NEEDS_REVIEW = "Needs Review"
    URGENT = "Urgent"


class SeverityLevel(StringEnum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class ConfidenceLevel(StringEnum):
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
