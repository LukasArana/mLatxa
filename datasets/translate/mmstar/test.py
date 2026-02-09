import re

def check_output_format(output):
    # This is the simplest "skeleton" check
    pattern = r"Aukerak:.*?A:.*?B:.*?C:.*?D:.*"

    # re.DOTALL is crucial so that '.*' matches newlines
    # re.IGNORECASE makes it catch 'aukerak' or 'AUKERAK'
    match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)

    return match

print(check_output_format('Nola daude irudiko pertsonak? Aukerak: A: inplikatuta, B: larrituta, C: haserre,  D: maitekor'))