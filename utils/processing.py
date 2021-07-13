def fill_placeholders(string, placeholder_to_value):
    for placeholder, value in placeholder_to_value.items():
        string = string.replace(placeholder, str(value))
    return string
