"""Regression tests for `parse_two_part_question`.

The parser splits rubric questions that bundle a broader context (Part 1)
with a specific focus (Part 2). The analyzer must drive its answer from
Part 2, so a parser miss silently downgrades the prompt to a generic
single-question template — losing the Part-2 focus.

These cases lock in the historical bugs:
- Pattern 2 was gated on `[A-ZSÉA]`, so lowercase / Á / Í / Ó / Ú starts
  silently fell through to `is_two_part=False`.
- Single-clause questions sometimes got artificially split.

Run with: `pytest test_two_part_parsing.py -v`
"""
from __future__ import annotations

import re

import pytest


def parse_two_part_question(question):
    """Mirror of the parser shipped in oli_v6_deploy.py / oli_v6_deploy_core.py.

    Kept inline so the test can run without importing the Streamlit app
    (which has heavy module-level side effects). Keep this in sync with
    the production copies.
    """
    if not isinstance(question, str) or question.count('?') < 2:
        return {'is_two_part': False, 'part1': None, 'part2': None, 'full_question': question}

    keywords = ['específicamente', 'en particular', 'en concreto', 'puntualmente', 'además']
    for keyword in keywords:
        pattern = rf'(.+?\?)\s*{keyword}[,:]?\s*¿\s*(.+?\?)'
        match = re.search(pattern, question, re.IGNORECASE | re.UNICODE)
        if match:
            return {
                'is_two_part': True,
                'part1': match.group(1).strip(),
                'part2': '¿' + match.group(2).strip(),
                'full_question': question,
            }

    match = re.search(r'(.+?\?)\s*¿\s*(.+\?)', question, re.UNICODE)
    if match:
        return {
            'is_two_part': True,
            'part1': match.group(1).strip(),
            'part2': '¿' + match.group(2).strip(),
            'full_question': question,
        }

    match = re.search(r'(.+?\?)(\S.+\?)', question, re.UNICODE)
    if match:
        part2 = match.group(2).strip()
        if not part2.startswith('¿'):
            part2 = '¿' + part2
        return {
            'is_two_part': True,
            'part1': match.group(1).strip(),
            'part2': part2,
            'full_question': question,
        }

    return {'is_two_part': False, 'part1': None, 'part2': None, 'full_question': question}


# ---------------------------------------------------------------------------
# Positive cases — these MUST be detected as two-part.
# ---------------------------------------------------------------------------

POSITIVE_CASES = [
    pytest.param(
        "¿Se definen los indicadores? Específicamente, ¿se incluyen metas cuantificables?",
        "¿Se definen los indicadores?",
        "¿se incluyen metas cuantificables?",
        id="keyword_separator_específicamente",
    ),
    pytest.param(
        "¿El proyecto considera aspectos de género?¿Se mencionan acciones específicas para la igualdad?",
        "¿El proyecto considera aspectos de género?",
        "¿Se mencionan acciones específicas para la igualdad?",
        id="explicit_¿_separator_capital_S",
    ),
    # REGRESSION: previously failed because Part 2 starts with lowercase.
    pytest.param(
        "¿Define el documento la teoría del cambio? ¿se identifican los supuestos clave?",
        "¿Define el documento la teoría del cambio?",
        "¿se identifican los supuestos clave?",
        id="regression_lowercase_start",
    ),
    # REGRESSION: previously failed because Part 2 starts with Á (not in [A-ZÉ]).
    pytest.param(
        "¿Existe un plan? ¿Áreas de mejora se identifican explícitamente?",
        "¿Existe un plan?",
        "¿Áreas de mejora se identifican explícitamente?",
        id="regression_accented_capital_Á",
    ),
    # REGRESSION: previously failed for Í / Ó / Ú starts.
    pytest.param(
        "¿Hay enfoque participativo? ¿Índices de participación están documentados?",
        "¿Hay enfoque participativo?",
        "¿Índices de participación están documentados?",
        id="regression_accented_capital_Í",
    ),
    pytest.param(
        "¿Se considera el enfoque de género? ¿Órganos rectores aprueban los indicadores?",
        "¿Se considera el enfoque de género?",
        "¿Órganos rectores aprueban los indicadores?",
        id="regression_accented_capital_Ó",
    ),
    pytest.param(
        "¿El presupuesto está claro? ¿Últimos cambios se documentaron?",
        "¿El presupuesto está claro?",
        "¿Últimos cambios se documentaron?",
        id="regression_accented_capital_Ú",
    ),
    # REGRESSION: previously failed for Ñ start.
    pytest.param(
        "¿Existen indicadores? ¿Ñoñerías aparte, se priorizan los críticos?",
        "¿Existen indicadores?",
        "¿Ñoñerías aparte, se priorizan los críticos?",
        id="regression_capital_Ñ",
    ),
    # REGRESSION: previously failed for digit start.
    pytest.param(
        "¿Hay metas anuales? ¿2026 incluye objetivos cuantificables?",
        "¿Hay metas anuales?",
        "¿2026 incluye objetivos cuantificables?",
        id="regression_digit_start",
    ),
    # The original rubric example shipped in the project — run-together with no separator.
    pytest.param(
        (
            "6.1 ¿La teoría del cambio (por ejemplo, la expectativa de cómo ocurrirá el cambio) está claramente "
            "expresada y expuesta explícitamente de una manera que parezca plausible para el no especialista, en "
            "la narrativa y / o como un gráfico?Se identifican los supuestos clave, incluidos los resultados de "
            "intervenciones pasadas, en curso o planificadas (incluidos los de otros actores, según corresponda)?"
        ),
        (
            "6.1 ¿La teoría del cambio (por ejemplo, la expectativa de cómo ocurrirá el cambio) está claramente "
            "expresada y expuesta explícitamente de una manera que parezca plausible para el no especialista, en "
            "la narrativa y / o como un gráfico?"
        ),
        (
            "¿Se identifican los supuestos clave, incluidos los resultados de intervenciones pasadas, en curso o "
            "planificadas (incluidos los de otros actores, según corresponda)?"
        ),
        id="run_together_no_separator",
    ),
]


@pytest.mark.parametrize("question,expected_part1,expected_part2", POSITIVE_CASES)
def test_detects_two_part(question, expected_part1, expected_part2):
    result = parse_two_part_question(question)
    assert result['is_two_part'] is True, f"Parser missed two-part structure: {question!r}"
    assert result['part1'] == expected_part1
    assert result['part2'] == expected_part2


# ---------------------------------------------------------------------------
# Negative cases — these must NOT be flagged as two-part.
# ---------------------------------------------------------------------------

NEGATIVE_CASES = [
    pytest.param("¿El documento incluye objetivos claros?", id="single_question"),
    pytest.param("Sin signo de pregunta.", id="no_question_marks"),
    pytest.param("¿X o Y o Z?", id="single_question_with_alternatives"),
    pytest.param("", id="empty_string"),
    pytest.param(None, id="none_input"),
]


@pytest.mark.parametrize("question", NEGATIVE_CASES)
def test_single_question_not_split(question):
    result = parse_two_part_question(question)
    assert result['is_two_part'] is False, f"Parser falsely split: {question!r}"
    assert result['part1'] is None
    assert result['part2'] is None


# ---------------------------------------------------------------------------
# Structural invariants.
# ---------------------------------------------------------------------------

def test_part2_always_starts_with_inverted_question_mark():
    cases = [c.values[0] for c in POSITIVE_CASES]
    for q in cases:
        result = parse_two_part_question(q)
        assert result['part2'].startswith('¿'), f"Part 2 missing leading ¿ for: {q!r}"


def test_full_question_is_preserved():
    q = "¿Hay plan? ¿se ejecuta?"
    result = parse_two_part_question(q)
    assert result['full_question'] == q
