"""
Test script to demonstrate two-part question parsing
"""
import re

def parse_two_part_question(question):
    """
    Detect and parse two-part rubric questions where:
    - Part 1 (broader): Sets the general context
    - Part 2 (specific): Asks the critical detail that needs primary focus

    Returns dict with 'is_two_part', 'part1', 'part2', 'full_question'
    """
    # Pattern 1: Questions with explicit keywords separating them (check this first)
    # Keywords like "Específicamente" act as clear separators
    keywords = ['específicamente', 'en particular', 'en concreto', 'puntualmente', 'además']
    for keyword in keywords:
        # The keyword is a separator, not part of either question
        pattern = rf'(.*?\?)\s*{keyword}[,:]?\s*¿\s*(.*?\?)'
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            part2 = match.group(2).strip()
            # Ensure part2 starts with ¿
            if not part2.startswith('¿'):
                part2 = '¿' + part2
            return {
                'is_two_part': True,
                'part1': match.group(1).strip(),
                'part2': part2,
                'full_question': question
            }

    # Pattern 2: Questions run together like "...gráfico?Se identifican..." or "...?¿Se..."
    # This is the most common pattern in the rubrics (no keyword separator)
    match = re.search(r'(.*?\?)\s*¿?\s*([A-ZSÉA].*?\?)', question)
    if match:
        part1 = match.group(1).strip()
        part2 = match.group(2).strip()

        # Ensure part2 starts with ¿ if it doesn't already
        if not part2.startswith('¿'):
            part2 = '¿' + part2

        return {
            'is_two_part': True,
            'part1': part1,
            'part2': part2,
            'full_question': question
        }

    # Not a two-part question
    return {
        'is_two_part': False,
        'part1': None,
        'part2': None,
        'full_question': question
    }


# Test with the provided example
test_question = "6.1 ¿La teoría del cambio (por ejemplo, la expectativa de cómo ocurrirá el cambio) está claramente expresada y expuesta explícitamente de una manera que parezca plausible para el no especialista, en la narrativa y / o como un gráfico?Se identifican los supuestos clave, incluidos los resultados de intervenciones pasadas, en curso o planificadas (incluidos los de otros actores, según corresponda)?"

print("=" * 80)
print("TESTING TWO-PART QUESTION PARSING")
print("=" * 80)
print("\nOriginal Question:")
print(test_question)
print("\n" + "=" * 80)

result = parse_two_part_question(test_question)

print("\nParsing Result:")
print(f"Is Two-Part: {result['is_two_part']}")
print("\n" + "-" * 80)
print("PART 1 (Broader Context):")
print(result['part1'])
print("\n" + "-" * 80)
print("PART 2 (Specific Focus - PRIMARY):")
print(result['part2'])
print("\n" + "=" * 80)

# Test additional examples
print("\n\nADDITIONAL TEST CASES:")
print("=" * 80)

test_cases = [
    "¿El documento incluye objetivos claros?",
    "¿Se definen los indicadores? Específicamente, ¿se incluyen metas cuantificables?",
    "¿El proyecto considera aspectos de género?¿Se mencionan acciones específicas para la igualdad?",
]

for i, case in enumerate(test_cases, 1):
    print(f"\nTest Case {i}:")
    print(f"Question: {case}")
    result = parse_two_part_question(case)
    print(f"Two-Part: {result['is_two_part']}")
    if result['is_two_part']:
        print(f"  Part 1: {result['part1']}")
        print(f"  Part 2: {result['part2']}")
    print("-" * 80)
