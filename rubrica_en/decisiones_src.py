# -*- coding: utf-8 -*-
"""English display text for «Rúbrica — Parcial», «— No» and «— No aplica».

Display only; the Spanish columns remain authoritative. Logical symbols
(∧ ∨ ¬ ≥ ≤) are preserved exactly — only the prose is translated.
"""

PARCIAL = {
"1.1.1": "DECISION: (T1 ∧ T2 ∧ T3) ∧ ¬T4   (cites everything but does not articulate contribution)\n    OR    (exactly 2 of T1/T2/T3) ∧ T4",
"1.1.2": "DECISION: T1 ∧ T2 ∧ ¬T3   (acknowledges the label but the level falls short)",
"1.1.3": "DECISION: T1 ∧ T2 ∧ ¬T3",
"1.2.1": "DECISION: (T1 ∨ T2) ∧ ¬T3   (names a framework but does not articulate the fit)",
"1.2.2": "DECISION: T1 ∧ ¬(T2 ∧ T3)",
"1.2.3": "DECISION: T1 ∧ ¬T2   (cites codes but they are not integrated into M&E)",
"1.3.1": "DECISION: T1 ∧ (T2 ∨ T3 ∨ T4)   but ¬ (T2 ∧ (T3 ∨ T4))",
"1.3.2": "DECISION: T1 ∧ ¬T3   OR   ¬T1 ∧ T3",
"1.3.3": "DECISION: T1 ∧ T2 ∧ ¬T3   (T2 is present but disaggregated data are missing)\n    OR    T1 ∧ T3 ∧ generic-T2 (T2 without specificity to the context)",
"1.3.4": "DECISION: T1 ∧ T2 ∧ ¬(T3 ∧ T4)",
"1.3.5": "DECISION: T1 ∧ T2 ∧ ¬T3   OR   T1 ∧ ¬T2",
"1.4.1": "DECISION: T1 ∧ T2 ∧ ¬T3",
"1.4.2": "DECISION: T1 ∨ T2 (only one is met)",
"1.4.3": "DECISION: T1 ∧ ¬T2   (cites but does not link)",
"1.5.1": "DECISION: (#true of T1/T2/T3) = 1   OR   only T4/T5 without T1/T2/T3",
"1.5.2": "DECISION: T1 ∧ ¬T2   (cited in the background but not integrated into the strategy)",
"1.5.3": "DECISION: (T1 ∨ T2) ∧ ¬T3   (cites observations without linking them to the strategy)",
"1.5.4": "DECISION: T1 ∧ exactly 1 of T2/T3/T4   OR   T5 absent where third parties are engaged",
"1.5.5": "DECISION: T1 ∧ ¬T2   OR   T1 ∧ T2 ∧ ¬T5 (where T5 applies)",
"1.5.6": """DECISION: (T1 ∨ T3) ∧ ¬(T1 ∧ T2 ∧ T3)

FLOOR RULE (T3): if T3 is TRUE, the result can NEVER be "No".
With T3 true, the result is "Yes" only if T1 and T2 are also met; in any
other case the result is, at minimum, "Partial".

Rationale: T3 evidences DEDICATED actions to transform gender relations.
That substance prevails over nomenclature. A PRODOC that carries out
dedicated transformative actions cannot be rated "No" for failing to label
them with the sensitive/responsive/transformative taxonomy (T1 false), nor
for failing to spell out the mechanism by which norms are challenged
(T2 false).

Cases this rule sends to "Partial":
- T3 true and T1 false: dedicated actions exist, without distinguishing the approach.
- T3 true and T2 false: dedicated actions exist, without articulating how they
  challenge norms or power relations.
- T1 true and ¬(T2 ∧ T3): the approach is declared but the actions remain
  numerical or merely inclusive.""",
"2.1.1": "DECISION: T1 ∧ T2 ∧ ¬(T3 ∧ T4)",
"2.1.2": "DECISION: T1 ∧ (T2 ∨ T3 ∨ T4) but not all three for all stakeholders",
"2.1.3": "DECISION: (T1 ∨ T2) ∧ ¬T3   OR   T1 ∧ T2 ∧ ¬T3",
"2.2.1": "DECISION: T1 ∧ ¬T2   OR   only claims of ownership without commitments",
"2.2.2": "DECISION: (T1 ∨ T2 ∨ T3) ∧ ¬T4   (asserted without documentary support)",
"2.2.3": "DECISION: T1 ∧ T2 ∧ (T3 ∨ T4) but not both",
"2.3.1": "DECISION: T1 ∧ T2 ∧ ¬T3",
"2.3.2": "DECISION: 2 of T1/T2/T3   OR   all 3 but with an almost exclusive emphasis on T1 (individual training)",
"2.4.1": "DECISION: T1 ∧ ¬T2   (a statement without operationalisation)",
"2.4.2": "DECISION: ¬T1 ∧ any of T2/T3/T4/T5   (scattered elements without an integrated plan)",
"2.4.3": "DECISION: T1 ∧ T2 ∧ T3 ∧ ¬(T4 ∧ T5)",
"2.4.4": "DECISION: (T1 ∨ T2) ∧ ¬T3",
"2.4.5": "DECISION: T1 ∧ T2 ∧ ¬(T3 ∧ T4)",
"3.1.1": "DECISION: T1 ∧ T2 ∧ T3 ∧ ¬T4   (a clear chain but the focus is on activities)",
"3.1.2": "DECISION: partial traceability exists but there are gaps in ≥1 outcome",
"3.1.3": "DECISION: T1 ∧ T2 ∧ ¬T3",
"3.1.4": "DECISION: T1 ∧ ¬T2   (generic assumptions)",
"3.1.5": "DECISION: T1 ∧ ¬(T2 ∧ T3)",
"3.2.1": "DECISION: T1 ∧ ¬T2   (a mix of change and action language)",
"3.2.2": "DECISION: outcomes meet 3–4 SMART attributes; typically M or T is missing",
"3.2.3": "DECISION: T1 ∧ ¬T2   (mixes outputs and activities)",
"3.3.1": "DECISION: appears only in assumptions or in the framework's cross-cutting language; no DEDICATED element",
"3.3.2": "DECISION: mentioned only at the level of assumptions or cross-cutting language; no DEDICATED element",
"3.4.1": "DECISION: T1 ∧ ¬T2",
"3.4.2": "DECISION: T1 ∧ T2 ∧ ¬(T3 ∧ T4)",
"3.4.3": "DECISION: covers only risks to staff, not to the community; or T1∧T2 ∧ ¬T4",
"3.4.4": "DECISION: T1 ∧ ¬T3",
"3.4.5": "DECISION: T1 ∧ a register is present but not the current version, or not attached though referenced",
"3.4.6": "DECISION: some risks specific, others generic; or the rating is missing",
"3.4.7": "DECISION: T1 universal but T2 or T3 missing for several risks",
"3.5.1": "DECISION: T1 ∧ ¬T2",
"3.5.2": "DECISION: T1 ∧ T3 ∧ ¬(T2 ∧ T4)",
"3.5.3": "DECISION: T1 ∧ T2 ∧ ¬(T3 ∧ T4)",
"3.5.4": "DECISION: T1 ∧ T4 ∧ T5 ∧ T6 but T2/T3 are generic (inclusive in language without real disaggregation)",
"3.5.5": "DECISION: T1 ∧ ¬T2   OR   an identifiable amount without a separate budget line",
"3.6.1": "DECISION: T3 ∧ ¬T2   (a timetable without justification)",
"3.6.2": "DECISION: T1 ∧ ¬T2",
"3.6.3": "DECISION: the need is acknowledged but left as \"if required\", with no budget line",
"3.7.1": "DECISION: T1 ∧ ¬T2",
"3.7.2": "DECISION: T2 ∧ ¬T1   (justifies without comparing alternatives)",
"4.1.1": "DECISION: T1 ∧ ¬T3 (centralisation without justification)",
"4.1.2": "DECISION: T1 ∧ ¬(T2 ∧ T3)",
"4.1.3": "DECISION: T1 ∧ ¬T2",
"4.1.4": "DECISION: 2–3 of T1/T2/T3/T4",
"4.1.5": "DECISION: T1 ∧ T2 ∧ ¬(T3 ∧ T4)",
"4.2.1": "DECISION: T1 ∧ T2 ∧ ¬T3",
"4.2.2": "DECISION: T1 ∧ ¬T2",
"4.2.3": "DECISION: T1 ∧ ¬T2   (mentioned without method or findings)",
"4.2.4": "DECISION: T1 ∧ T2 ∧ ¬T3   (gender remains at framing level)",
"4.2.5": "DECISION: T1 ∧ ¬T2   OR   track record absent but the partners are well established",
"4.3.1": "DECISION: T1 ∧ T2 ∧ ¬(T3 ∧ T4)",
"4.4.1": "DECISION: announces a strategy without a timetable or products",
"4.4.2": "DECISION: T1 ∧ T2 ∧ ¬T3   (accessibility declared without a budget line)",
"4.4.3": "DECISION: T1 ∧ T2 ∧ T3 ∧ ¬T4",
"5.1.1": "DECISION: 2 of T1/T2/T3",
"5.2.1": "DECISION: mostly clear text but with dense or repetitive sections",
}

NO = {
"1.1.1": "DECISION: (≤1 of T1/T2/T3 true)   OR   no detectable contribution verb",
"1.1.2": "DECISION: ¬T1   OR   ¬T2 (FRAMING only, nothing DEDICATED)",
"1.1.3": "DECISION: ¬T1   OR   marker ≥2 ∧ nothing DEDICATED",
"1.2.1": "DECISION: ¬T1 ∧ ¬T2   (names no specific framework)",
"1.2.2": "DECISION: ¬T1   (only generic consistency with the UNSDCF)",
"1.2.3": "DECISION: ¬T1   (names only SDGs \"5\", \"8\" without indicator codes)",
"1.3.1": "DECISION: ¬T1   OR   T1 ∧ ¬T2 ∧ ¬T3",
"1.3.2": "DECISION: ¬T1 ∧ ¬T3",
"1.3.3": "DECISION: T1 ∧ ¬T2",
"1.3.4": "DECISION: ¬T1   OR   ¬T2 (a list only, without analysis)",
"1.3.5": "DECISION: ¬T1   OR   consultation only with authorities/partners, without marginalised groups",
"1.4.1": "DECISION: ¬T1 ∨ ¬T2",
"1.4.2": "DECISION: ¬T1 ∧ ¬T2",
"1.4.3": "DECISION: ¬T1   OR   only \"lessons have been taken into account\" without a citation",
"1.5.1": "DECISION: (#true of T1/T2/T3/T4/T5) = 0",
"1.5.2": "DECISION: ¬T1   (does not cite conventions by number)",
"1.5.3": "DECISION: ¬T1 ∧ ¬T2",
"1.5.4": "DECISION: ¬T1   OR   only a general statement with no operational elements",
"1.5.5": "DECISION: ¬T1   (decorative mentions only)",
"1.5.6": """DECISION: ¬T1 ∧ ¬T3

"No" requires BOTH conditions to hold: the document does not distinguish the
type of approach (T1 false) AND it presents no dedicated actions to transform
gender relations (T3 false).

If T3 is true, "No" is ruled out by the FLOOR RULE (see Partial), whatever the
values of T1 and T2.

Typical "No" case: the document only states that it "mainstreams gender", or
uses generic equality formulations, with no dedicated actions and no
distinction of approach.""",
"2.1.1": "DECISION: ¬T1 ∨ ¬T2",
"2.1.2": "DECISION: ¬T1   (a list only, without analysis)",
"2.1.3": "DECISION: ¬T1 ∧ ¬T2",
"2.2.1": "DECISION: ¬T1",
"2.2.2": "DECISION: ¬T1 ∧ ¬T2 ∧ ¬T3",
"2.2.3": "DECISION: ¬T1",
"2.3.1": "DECISION: ¬T1   OR   only a vague statement, \"capacities will be assessed\"",
"2.3.2": "DECISION: T1 only   (reduces capacity development to workshops)",
"2.4.1": "DECISION: ¬T1",
"2.4.2": "DECISION: ¬T1 ∧ no operational element",
"2.4.3": "DECISION: T1 ∧ ¬T2 ∧ ¬T3",
"2.4.4": "DECISION: ¬T1 ∧ ¬T2",
"2.4.5": "DECISION: ¬T1",
"3.1.1": "DECISION: ¬T1   (the logic can only be inferred from the activity list)",
"3.1.2": "DECISION: pervasive gaps; outputs and outcomes are decoupled",
"3.1.3": "DECISION: only \"capacities will be developed\", without specifying",
"3.1.4": "DECISION: ¬T1   OR   assumptions confused with risks",
"3.1.5": "DECISION: ¬T1",
"3.2.1": "DECISION: ¬T1   (action language predominates)",
"3.2.2": "DECISION: outcomes are aspirations (≤2 attributes)",
"3.2.3": "DECISION: ¬T1",
"3.3.1": "DECISION: the logical framework mentions gender neither in outcomes, outputs nor indicators",
"3.3.2": "DECISION: the logical framework does not include disability",
"3.4.1": "DECISION: ¬T1",
"3.4.2": "DECISION: states only \"zero tolerance\", without operationalisation",
"3.4.3": "DECISION: ¬T1",
"3.4.4": "DECISION: ¬T1",
"3.4.5": "DECISION: T1 ∧ ¬T2",
"3.4.6": "DECISION: the risks are institutional boilerplate, without contextualisation",
"3.4.7": "DECISION: medium/high risks with no measures, or only generic measures",
"3.5.1": "DECISION: \"monitoring will be carried out\", with no system described",
"3.5.2": "DECISION: ¬T1   (data are collected with no defined use)",
"3.5.3": "DECISION: ¬T1   OR   only an indicator table, without methods",
"3.5.4": "DECISION: indicators are not SMART, lack a baseline, or ignore gender/disability",
"3.5.5": "DECISION: ¬T1   (no budget line and no identifiable amount)",
"3.6.1": "DECISION: ¬T3",
"3.6.2": "DECISION: budget only at the level of broad categories",
"3.6.3": "DECISION: no budget lines and no recognition of cross-cutting costs",
"3.7.1": "DECISION: ¬T1",
"3.7.2": "DECISION: ¬T1 ∧ ¬T2",
"4.1.1": "DECISION: ¬T1",
"4.1.2": "DECISION: only \"staff will be recruited as required\"",
"4.1.3": "DECISION: assumes support without provision or identification",
"4.1.4": "DECISION: only \"ILO procedures will apply\"",
"4.1.5": "DECISION: only \"third parties will comply with the standards\"",
"4.2.1": "DECISION: lists partners only, without roles or justification",
"4.2.2": "DECISION: ¬T1",
"4.2.3": "DECISION: ¬T1",
"4.2.4": "DECISION: gaps identified but no plan",
"4.2.5": "DECISION: merely asserts that they will deliver well",
"4.3.1": "DECISION: \"stakeholders will be informed\", without operationalisation",
"4.4.1": "DECISION: no public communication strategy is included",
"4.4.2": "DECISION: ¬T1 ∧ ¬T2",
"4.4.3": "DECISION: T1 ∧ T2 ∧ ¬T3 ∧ ¬T4",
"5.1.1": "DECISION: key sections or annexes are missing",
"5.2.1": "DECISION: core ideas are buried; hard to follow",
}

if __name__ == "__main__":
    import json
    import pathlib
    here = pathlib.Path(__file__).parent
    for name, data in (("tests_parcial.json", PARCIAL), ("tests_no.json", NO)):
        (here / name).write_text(
            json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"{len(data):3} entradas -> {name}")
