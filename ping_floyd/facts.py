"""Project a sourced fact into the requested representation without guessing."""
from datetime import date
import re


def project_fact(value, requested):
    value = str(value).strip()
    if requested != 'birth_year':
        return value
    if re.fullmatch(r'\d{4}',value):
        return value
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}',value):
        try:
            return str(date.fromisoformat(value).year)
        except ValueError:
            pass
    return None
