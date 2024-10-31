def clean_markdown_content(file_path):
    # Открываем файл и читаем содержимое
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return remove_initial_headers_until_text(content)

def remove_initial_headers_until_text(text):
    lines = text.splitlines()
    header_lines = []
    for line in lines:
        if (line.startswith("#") or line.startswith('\n') or not line or line.startswith('**')):
            header_lines.append(line)
        else:
            break
    for line in header_lines:
        lines.remove(line)
    return "\n".join([header_lines[0]] + [header_lines[-1]] + lines).strip()