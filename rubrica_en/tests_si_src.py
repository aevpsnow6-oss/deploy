# -*- coding: utf-8 -*-
"""English display text for «Rúbrica — Sí» (the TESTS block).

Faithful translation of the Spanish, which remains authoritative: this text
is shown to the reader, never sent to the model. Structure must be kept —
the renderer parses «Tn:» labels and the «DECISION:» line.

Run rubrica_en/apply.py after editing.
"""

TESTS = {
"1.1.1": """TESTS:
T1: Does it cite at least one P&B outcome/priority by code or name? (yes/no)
T2: Does it cite at least one DWCP outcome with the country name? (yes/no)
T3: Does it cite at least one CPO by code (letter+number format, e.g. ABC-101)? (yes/no)
T4: Does the text link the intervention to T1/T2/T3 with a CONTRIBUTION verb whose target is one of the results cited in T1, T2 or T3? (yes/no)
    Counts: "contributes to", "feeds into", "translates into", "supports the achievement of", and their equivalents in the language of the document.
    Does NOT count: verbs of mere alignment or consistency — "aligned with", "consistent with", "in line with", "responds to" — because they declare correspondence, not contribution.
    Nor does a contribution verb count when its target is a generic objective (e.g. "will contribute to more sustainable outcomes in the country") or an entity not cited in T1/T2/T3: the target must be the identified P&B, DWCP or CPO result.

DECISION: T1 ∧ T2 ∧ T3 ∧ T4""",

"1.1.2": """TESTS:
T1: Does the proposal name the CPO disability label/marker (literal text: "label", "marker", "principal", "significant", "limited")? (yes/no)
T2: Is there ≥1 DEDICATED disability element (sub-objective / indicator / activity / budget line / target)? (yes/no)
T3: Does the number of DEDICATED elements match the label (principal→≥3 DEDICATED; significant→≥1; limited→0–1)? (yes/no)

FILTER: only DEDICATED elements count, not FRAMING (lists of ≥3 groups, "among others", inclusive boilerplate).

DECISION: T1 ∧ T2 ∧ T3""",

"1.1.3": """TESTS:
T1: Does the proposal name the CPO gender marker (text: "marker 0/1/2/3", "GEM 0–3", "principal", "significant")? (yes/no)
T2: Is there ≥1 DEDICATED gender element (sub-objective / disaggregated indicator / activity / budget line / target)? (yes/no)
T3: Does the number of DEDICATED elements match the marker (3→≥3 DEDICATED; 2→≥2; 1→≥1)? (yes/no)

FILTER: DEDICATED vs FRAMING as in 1.1.2.

DECISION: T1 ∧ T2 ∧ T3""",

"1.2.1": """TESTS:
T1: Does it name the country's national development plan by title (not a generic "the national plan")? (yes/no)
T2: Does it name the country's current UNSDCF / UNDAF? (yes/no)
T3: Does it articulate how the project fits (verb + object) with at least one of the frameworks in T1/T2? (yes/no)

DECISION: (T1 ∨ T2) ∧ T3""",

"1.2.2": """TESTS:
T1: Does it cite specific UNSDCF outcomes (by number or full title)? (yes/no)
T2: Does it identify the areas where the ILO is lead or convening agency? (yes/no)
T3: Does it articulate the project's contribution to the outcomes in T1? (yes/no)

DECISION: T1 ∧ T2 ∧ T3""",

"1.2.3": """TESTS:
T1: Does it cite ≥2 SDG indicators by code (N.N.N format, such as 8.5.2, 1.3.1)? (yes/no)
T2: Do those indicators appear in the logical framework or the M&E plan? (yes/no)

DECISION: T1 ∧ T2""",

"1.3.1": """TESTS:
T1: Is there a section/annex labelled "situation analysis" or "problem statement"? (yes/no)
T2: Does it cite ≥2 verifiable sources (studies, official data, evaluations, censuses)? (yes/no)
T3: Does it quantify the magnitude (figure, percentage, number of people affected)? (yes/no)
T4: Does it bound the scope in time, geography or population? (yes/no)

DECISION: T1 ∧ T2 ∧ (T3 ∨ T4)""",

"1.3.2": """TESTS:
T1: Does the text distinguish at least 2 causal levels (immediate, underlying, structural) or use a problem tree? (yes/no)
T2: Are the causes supported by evidence or consultation (citation / reference)? (yes/no)
T3: Is the project strategy explicitly mapped to causes (not only to symptoms)? (yes/no)

DECISION: T1 ∧ T3""",

"1.3.3": """TESTS:
T1: Does it identify the population group by name and magnitude? (yes/no)
T2: Is there a gender analysis of the context (covering at least one of: roles, division of labour, differentiated opportunities/constraints)? (yes/no)
T3: Is sex-disaggregated data provided where available? (yes/no)

The DEDICATED vs FRAMING filter applies to the gender analysis (T2).

DECISION: T1 ∧ T2 ∧ T3""",

"1.3.4": """TESTS:
T1: Is there a stakeholder mapping (table/dedicated section)? (yes/no)
T2: Does each key stakeholder have both interests AND constraints stated? (yes/no)
T3: Does it include women's organizations by name? (yes/no)
T4: If the project affects persons with disabilities → does it include OPDs (organizations of persons with disabilities)? (yes/no/NA)

DECISION: T1 ∧ T2 ∧ T3 ∧ (T4 ∨ NA)""",

"1.3.5": """TESTS:
T1: Is there a consultation plan with a documented methodology? (yes/no)
T2: Does it identify concrete accessibility/equity measures (interpretation, scheduling, formats, local languages)? (yes/no)
T3: Is there an analysis of potential discriminatory effects and their mitigation? (yes/no)
T4: If it affects indigenous peoples → does it include an FPIC (Free, Prior and Informed Consent) process? (yes/no/NA)

DECISION: T1 ∧ T2 ∧ T3 ∧ (T4 ∨ NA)""",

"1.4.1": """TESTS:
T1: Explicit reference to the ILO's normative identity (conventions, recommendations, standards)? (yes/no)
T2: Explicit reference to tripartism (government + employers + workers named)? (yes/no)
T3: A value-added argument specific to this project (not institutional boilerplate)? (yes/no)

DECISION: T1 ∧ T2 ∧ T3""",

"1.4.2": """TESTS:
T1: Does it describe the ILO's presence in the country (office, team, length of presence)? (yes/no)
T2: Does it list ≥2 past or ongoing projects with title or code? (yes/no)

DECISION: T1 ∧ T2""",

"1.4.3": """TESTS:
T1: Does it cite ≥1 specific evaluation (title, year, project)? (yes/no)
T2: Does it link at least one lesson to a visible decision in the present design? (yes/no)
T3 (optional): Does it draw out concrete lessons, as a list or in prose? (yes/no)

DECISION: T1 ∧ T2""",

"1.5.1": """TESTS (DEDICATED vs FRAMING filter):
T1: A sub-objective / outcome / output that NAMES disability? (yes/no)
T2: An indicator disaggregated by disability, or disability-specific? (yes/no)
T3: An activity whose main purpose is disability? (yes/no)
T4 (optional): A budget line for disability? (yes/no)
T5 (optional): A quantifiable target relating to disability? (yes/no)

DECISION: (#true of T1/T2/T3) ≥ 2""",

"1.5.2": """TESTS:
T1: Does it cite ILO conventions/recommendations by number (C087, C098, C111, C190…)? (yes/no)
T2: Do those instruments appear in the strategy, indicators or objectives (not only in the background)? (yes/no)
T3: Does it include actions promoting ratification / application / awareness? (yes/no)
T4: Are there references to NORMLEX or another ILO legal source? (yes/no)

DECISION: T1 ∧ T2 ∧ (T3 ∨ T4)""",

"1.5.3": """TESTS:
T1: Does it cite CEACR observations on the country? (yes/no)
T2: Does it cite conclusions of the Committee on the Application of Standards or the Committee on Freedom of Association? (yes/no)
T3: Do the cited observations inform the justification or the strategy (explicit link)? (yes/no)

DECISION: (T1 ∨ T2) ∧ T3""",

"1.5.4": """TESTS:
T1: An explicit commitment to ILS compliance within the project? (yes/no)
T2: Fair wages specified (legal or sectoral reference)? (yes/no)
T3: OSH conditions applicable to staff and to contractors? (yes/no)
T4: An accessible grievance mechanism (channel + languages + confidentiality)? (yes/no)
T5 (cond): if third parties are engaged → compliance clauses in contracts? (yes/no/NA)

DECISION: T1 ∧ (#true of T2/T3/T4) ≥ 2 ∧ (T5 ∨ NA)""",

"1.5.5": """TESTS:
T1: An analysis of the project's potential environmental impacts (not generic)? (yes/no)
T2: Mitigation measures specific to the impacts identified? (yes/no)
T3: Does it consider biodiversity OR affected communities? (yes/no)
T4: Sustainable practices in implementation (materials / energy / waste)? (yes/no)
T5 (cond): if there is infrastructure/construction → specific environmental safety measures? (yes/no/NA)

DECISION: T1 ∧ T2 ∧ (#true of T3/T4) ≥ 1 ∧ (T5 ∨ NA)""",

"1.5.6": """TESTS:
T1: Does it distinguish the type of approach (gender-sensitive / responsive / transformative)? (yes/no)
T2: Does it articulate how the project challenges norms or power relations? (yes/no)
T3: DEDICATED actions to transform relations (not merely to include women)? (yes/no)
T4 (optional): Do indicators measure changes in relations (not only numerical participation)? (yes/no)

DECISION: T1 ∧ T2 ∧ T3""",
}

TESTS.update({
"2.1.1": """TESTS:
T1: Does it describe the process (what, when, with whom) of consultations during design? (yes/no)
T2: Does it list the constituents consulted (government + employers + workers named)? (yes/no)
T3: Does it define their role in project monitoring? (yes/no)
T4: Does it define their role in implementation? (yes/no)

DECISION: T1 ∧ T2 ∧ T3 ∧ T4""",

"2.1.2": """TESTS:
T1: Key stakeholders identified by name (not generically). (yes/no)
T2: Each stakeholder's link to the final beneficiaries. (yes/no)
T3: Role in relation to the problem (cause/affected/mitigator). (yes/no)
T4: Role in relation to the solution (implementer/validator/beneficiary). (yes/no)

DECISION: T1 ∧ T2 ∧ T3 ∧ T4 for ALL key stakeholders""",

"2.1.3": """TESTS:
T1: Explicit mention of consultation with ACT/EMP? (yes/no)
T2: Explicit mention of consultation with ACTRAV? (yes/no)
T3: Evidence that it was taken up (a decision, an adjustment, a section cited)? (yes/no)

DECISION: T1 ∧ T2 ∧ T3""",

"2.2.1": """TESTS:
T1: Does it describe the consultations held (when, with whom, on what)? (yes/no)
T2: Does it list concrete commitments made by partners? (yes/no)
T3 (optional): Are those commitments operational (time / resources / decisions)? (yes/no)

DECISION: T1 ∧ T2""",

"2.2.2": """TESTS:
T1: Evidence that each key partner has accepted the objectives? (yes/no)
T2: Acceptance of the performance framework / indicators? (yes/no)
T3: Acceptance of obligations and responsibilities? (yes/no)
T4: A supporting document referenced (MoU, minutes, letter)? (yes/no)

DECISION: T1 ∧ T2 ∧ T3 ∧ T4""",

"2.2.3": """TESTS:
T1: An explicit strategy for continuous ownership (not just one-off events)? (yes/no)
T2: Shared governance mechanisms (committees, steering groups)? (yes/no)
T3: A link to the post-project sustainability strategy? (yes/no)
T4: A strategy differentiated by type of actor (not uniform)? (yes/no)

DECISION: T1 ∧ T2 ∧ T3 ∧ T4""",

"2.3.1": """TESTS:
T1: A capacity assessment carried out OR planned, with a named methodology? (yes/no)
T2: Explicit conclusions (gaps against existing capacities)? (yes/no)
T3: Do those conclusions appear integrated into the project design? (yes/no)

DECISION: T1 ∧ T2 ∧ T3""",

"2.3.2": """TESTS:
T1: Individual (technical) capacities? (yes/no)
T2: Organizational capacities (systems, processes)? (yes/no)
T3: Enabling environment (legal framework, policies, financing)? (yes/no)

DECISION: T1 ∧ T2 ∧ T3""",

"2.4.1": """TESTS:
T1: Explicit evidence of constituents' interest in the post-project period? (yes/no)
T2: Is that evidence concrete (letter, MoU, allocation of future staff/budget)? (yes/no)

DECISION: T1 ∧ T2""",

"2.4.2": """TESTS:
T1: An EXPLICIT sustainability plan (section, annex or clear reference)? (yes/no)
T2 (adds quality): an institutional mechanism. (yes/no)
T3 (adds quality): a financial mechanism. (yes/no)
T4 (adds quality): a governance mechanism. (yes/no)
T5 (adds quality): a transition timetable. (yes/no)

DECISION: T1   (an explicit plan suffices; T2–T5 raise the score)""",

"2.4.3": """TESTS:
T1: Does the document EXPLICITLY define the pilot/test phase? (yes/no/NA)
T2: Does it identify the preconditions needed to scale up? (yes/no)
T3: Does it define objective success criteria? (yes/no)
T4: A plan for measuring/demonstrating success? (yes/no)
T5: A plan for transition to the scale-up phase? (yes/no)

DECISION: T1 ∧ T2 ∧ T3 ∧ T4 ∧ T5""",

"2.4.4": """TESTS:
T1: An identified source of human resources after the project? (yes/no)
T2: An identified source of financial resources after the project? (yes/no)
T3: A plausibility argument with support (existing sectoral budget, institutional commitment)? (yes/no)

DECISION: T1 ∧ T2 ∧ T3""",

"2.4.5": """TESTS:
T1: An EXPLICIT exit strategy (section or annex)? (yes/no)
T2: A plan for transferring responsibilities (what, to whom, when)? (yes/no)
T3: Capacity development actions specific to the transfer? (yes/no)
T4: An exit timetable with milestones? (yes/no)

DECISION: T1 ∧ T2 ∧ T3 ∧ T4""",

"3.1.1": """TESTS:
T1: Is the outputs → outcomes → impact causal chain made explicit (not implicit)? (yes/no)
T2: Are mechanisms of change identified (verb + subject)? (yes/no)
T3: Are critical assumptions stated as such (not as risks)? (yes/no)
T4: Ratio of results-text to activities-text > 1. (yes/no)

DECISION: T1 ∧ T2 ∧ T3 ∧ T4""",

"3.1.2": """TESTS:
T1: Coverage — the outputs cover the outcomes with no logical gaps. (yes/no)
T2: Sufficiency — each outcome has outputs that justify achieving it. (yes/no)
T3: The outcomes→impact link is reasoned. (yes/no)

DECISION: T1 ∨ T2 ∨ T3   (at least one; subjectivity acknowledged)""",

"3.1.3": """TESTS:
T1: Are the actors of capacity development named (not "the partners")? (yes/no)
T2: Is the type of change expected specified (performance / behaviour / practices)? (yes/no)
T3: Is the mechanism by which capacity → change articulated? (yes/no)

DECISION: T1 ∧ T2 ∧ T3""",

"3.1.4": """TESTS:
T1: Are assumptions stated with a conditional verb ("it is assumed that…")? (yes/no)
T2: Are assumptions tied to levels of the chain (outputs/outcomes/impact)? (yes/no)
T3: Is there reference to past or parallel interventions (ILO or others)? (yes/no)

DECISION: T1 ∧ T2 ∧ T3""",

"3.1.5": """TESTS:
T1: Does it cite specific evaluations (title, year)? (yes/no)
T2: Do those evaluations support causal links in the logical framework? (yes/no)
T3: Are lessons explicitly applied to the present design? (yes/no)

DECISION: T1 ∧ T2 ∧ T3""",

"3.2.1": """TESTS:
T1: Do the outcomes express states / situations ("X is strengthened", "Y has access")? (yes/no)
T2: Absence of activity formulations ("through", "by means of", "by carrying out") in >80% of the outcomes. (yes/no)

DECISION: T1 ∧ T2""",

"3.2.2": """SMART TESTS per outcome:
S — Specific (subject + verb + concrete object). (yes/no)
M — Measurable (indicator with a quantifiable threshold). (yes/no)
A — Achievable (feasible with the project's resources). (yes/no)
R — Relevant (linked to the problem and the ToC). (yes/no)
T — Time-bound (with a defined deadline or traceable to the timetable). (yes/no)

DECISION: ALL outcomes meet S+M+A+R+T""",

"3.2.3": """TESTS:
T1: Are outputs framed as deliverables ("Document X published", "N people certified")? (yes/no)
T2: Absence of activity formulations ("Hold workshops", "Train…") in >80% of the outputs. (yes/no)

DECISION: T1 ∧ T2""",

"3.3.1": """TESTS (DEDICATED vs FRAMING filter):
T1: An outcome or output whose title NAMES gender/equality/women? (yes/no)
T2: An indicator disaggregated by sex or gender-specific? (yes/no)
T3: A quantifiable target by sex or on gender? (yes/no)
T4 (cond): Is gender inclusion declared EXPLICIT in the project? (yes/no)

DECISION:
  if T4=yes → T1 ∧ (T2 ∨ T3)
  if T4=no  → T2 ∨ T3""",

"3.3.2": """TESTS (DEDICATED vs FRAMING filter):
T1: An outcome or output that NAMES disability? (yes/no)
T2: A disability-specific or disaggregated indicator? (yes/no)
T3: A quantifiable disability target? (yes/no)
T4 (cond): Is disability inclusion declared EXPLICIT in the project? (yes/no)

DECISION:
  if T4=yes → T1 ∧ (T2 ∨ T3)
  if T4=no  → T2 ∨ T3""",
})

TESTS.update({
"3.4.1": """TESTS:
T1: Is there a risk analysis section or annex? (yes/no)
T2: Are the risks categorised (strategic / operational / fiduciary / contextual / reputational)? (yes/no)

DECISION: T1 ∧ T2""",

"3.4.2": """TESTS:
T1: Are SEA-specific risks identified? (yes/no)
T2: Prevention mechanisms (code of conduct, mandatory training)? (yes/no)
T3: An accessible and confidential reporting mechanism? (yes/no)
T4: A response protocol and victim support? (yes/no)
T5 (cond): if third parties are engaged → SEA clauses in contracts? (yes/no/NA)

DECISION: T1 ∧ T2 ∧ T3 ∧ T4 ∧ (T5 ∨ NA)""",

"3.4.3": """TESTS:
T1: Does it identify community risks (not only risks to project staff)? (yes/no)
T2: Measures for physical risks (traffic / hazardous materials / pollution)? (yes/no)
T3 (cond): if security personnel are engaged → a use-of-force and abuse-prevention protocol? (yes/no/NA)
T4: A community grievance mechanism? (yes/no)

DECISION: T1 ∧ T2 ∧ T4 ∧ (T3 ∨ NA)""",

"3.4.4": """TESTS:
T1: Does it identify ILS non-compliance risks in the project context? (yes/no)
T2 (optional): A link to CEACR observations or supervisory bodies? (yes/no)
T3: A specific mitigation plan? (yes/no)

DECISION: T1 ∧ T3   (T2 raises the score but is not required)""",

"3.4.5": """TESTS:
T1: Is the project budget > USD 1,000,000? (yes/no/NA)
T2: Is a risk register in the current ILO format attached? (yes/no)

DECISION: if T1=yes → T2; if T1=no → N/A""",

"3.4.6": """TESTS:
T1: Is each risk contextual to the project (not a generic template)? (yes/no)
T2: Are likelihood and impact rated (numerical or categorical scale)? (yes/no)
T3: Are the risks material to the project's success? (yes/no)

DECISION: T1 ∧ T2 ∧ T3""",

"3.4.7": """TESTS, for each medium- or high-level risk:
T1: A mitigation measure? (yes/no)
T2: An owner for the measure? (yes/no)
T3: A monitoring method? (yes/no)

DECISION: T1 ∧ T2 ∧ T3 for ALL medium/high risks""",

"3.5.1": """TESTS:
T1: Is the collection system described (what data, frequency, who is responsible)? (yes/no)
T2: A justification of the resources allocated to M&E (linked to the budget)? (yes/no)
T3 (cond): if the budget exceeds the threshold → an evaluability review? (yes/no/NA)
T4 (optional): A project learning plan? (yes/no)

DECISION: T1 ∧ T2 ∧ (T3 ∨ NA)""",

"3.5.2": """TESTS:
T1: Is the feedback loop defined (data → decisions)? (yes/no)
T2: Are the information needs for reporting specified? (yes/no)
T3: Are responsible staff named, or identified by post? (yes/no)
T4: Is the frequency of review cycles defined? (yes/no)

DECISION: T1 ∧ T2 ∧ T3 ∧ T4""",

"3.5.3": """TESTS:
T1: An M&E plan as a discrete document or section? (yes/no)
T2: Are collection methods named (survey, interview, administrative records)? (yes/no)
T3: Are analysis methods specified? (yes/no)
T4: Roles and responsibilities per method/indicator? (yes/no)

DECISION: T1 ∧ T2 ∧ T3 ∧ T4""",

"3.5.4": """TESTS per indicator in the M&E plan:
T1: Does it meet SMART in full (S+M+A+R+T)? (yes/no)
T2: Is it gender-sensitive (sex-disaggregated or gender-specific)? (yes/no)
T3: Does it allow for disability inclusion (disaggregated or specific)? (yes/no)
T4: Is a baseline defined? (yes/no)
T5: Are a target and milestones defined? (yes/no)
T6: Does it cover outcomes (not only outputs)? (yes/no)

DECISION: ALL indicators satisfy T1+T2+T3+T4+T5+T6""",

"3.5.5": """TESTS:
T1: Is the evaluation budget line SEPARATE and named? (yes/no)
T2: Is the amount ≥ ~2% of the total budget (or is a departure justified)? (yes/no)

DECISION: T1 ∧ T2""",

"3.6.1": """TESTS:
T1 (cond): if there is an inception period → are its activities and results made explicit? (yes/no/NA)
T2: Is the overall timeframe justified against complexity and capacities? (yes/no)
T3: A timetable with quarterly or half-yearly milestones? (yes/no)

DECISION: T2 ∧ T3 ∧ (T1 ∨ NA)""",

"3.6.2": """TESTS:
T1: Is the budget broken down by activity or output? (yes/no)
T2: Is there consistency between logical framework activities and budget lines? (yes/no)
T3 (optional): Are unit costs or calculations visible? (yes/no)

DECISION: T1 ∧ T2""",

"3.6.3": """TESTS (check those applicable to the project):
T1: A budget line for a gender specialist where applicable? (yes/no/NA)
T2: A budget line for a disability specialist where applicable? (yes/no/NA)
T3: A budget line for information accessibility (plain language, alternative formats)? (yes/no/NA)
T4: A budget line for interpretation into local languages or sign language where applicable? (yes/no/NA)

DECISION: ALL applicable items (T1–T4) have a specific budget line with an amount""",

"3.7.1": """TESTS:
T1: An explicit mention of cost-effectiveness or "value for money"? (yes/no)
T2: Is it linked to the project design (not only to administration)? (yes/no)

DECISION: T1 ∧ T2""",

"3.7.2": """TESTS:
T1: An analysis of design alternatives with costs/benefits? (yes/no)
T2: A justification of the option chosen? (yes/no)
T3 (cond): A comparison against benchmarks (similar projects) where data exist? (yes/no/NA)

DECISION: T1 ∧ T2 ∧ (T3 ∨ NA)""",

"4.1.1": """TESTS:
T1: Does it identify the managing office/unit by name? (yes/no)
T2 (cond): if the project is regional/national → is it a field office close to the beneficiaries? (yes/no/NA)
T3 (cond): if managed centrally from headquarters → is there an explicit justification on effectiveness/cost-effectiveness/capacity? (yes/no/NA)

DECISION: T1 ∧ (T2 ∨ T3)""",

"4.1.2": """TESTS:
T1: An organigram or description of staff roles? (yes/no)
T2: An explicit line of accountability (who reports to whom)? (yes/no)
T3: The responsible ILO official identified by post or name? (yes/no)

DECISION: T1 ∧ T2 ∧ T3""",

"4.1.3": """TESTS:
T1: Does it identify the technical or administrative units that will provide support? (yes/no)
T2: Is that support budgeted for (a line or allocation)? (yes/no)

DECISION: T1 ∧ T2""",

"4.1.4": """TESTS:
T1: A staffing plan? (yes/no)
T2: Procurement procedures (ILO standards or adaptations)? (yes/no)
T3: Financial and reporting systems? (yes/no)
T4: Levels of authority and approval? (yes/no)

DECISION: T1 ∧ T2 ∧ T3 ∧ T4""",

"4.1.5": """TESTS (where there are contractors/subcontractors/suppliers):
T1: Decent work clauses in third-party contracts? (yes/no)
T2: Fair employment clauses (wages, working time, freedom of association)? (yes/no)
T3: A grievance mechanism accessible to third-party workers? (yes/no)
T4: Compliance monitoring by the ILO? (yes/no)

DECISION: T1 ∧ T2 ∧ T3 ∧ T4""",

"4.2.1": """TESTS:
T1: Are the roles of each institution/partner made explicit? (yes/no)
T2: Implementation procedures (how they work together)? (yes/no)
T3: A justification for the choice of partners (why these and not others)? (yes/no)

DECISION: T1 ∧ T2 ∧ T3""",

"4.2.2": """TESTS:
T1: Explicit evidence of acceptance by each partner? (yes/no)
T2: A supporting document (letter, MoU, minutes) referenced? (yes/no)

DECISION: T1 ∧ T2""",

"4.2.3": """TESTS:
T1: An organizational capacity assessment carried out or planned, with a methodology? (yes/no)
T2: An explicit reference in the proposal (not only in an uncited annex)? (yes/no)

DECISION: T1 ∧ T2""",

"4.2.4": """TESTS (cond: where the assessment identified gaps):
T1: A capacity development plan for the implementing partners? (yes/no)
T2: Operational components (training, mentoring, accompaniment)? (yes/no)
T3: A gender-responsiveness component? (yes/no)

DECISION: T1 ∧ T2 ∧ T3""",

"4.2.5": """TESTS:
T1: Are the partners' track records documented (similar projects, demonstrated capacity)? (yes/no)
T2: A plausibility argument with evidence (not simply "they are competent")? (yes/no)
T3 (cond): if demonstrated capacity is limited → specific mitigation? (yes/no/NA)

DECISION: T1 ∧ T2 ∧ (T3 ∨ NA)""",

"4.3.1": """TESTS:
T1: A stakeholder communication plan (what, when, how)? (yes/no)
T2: Expected communication products (reports, briefings, events)? (yes/no)
T3: Human resources allocated? (yes/no)
T4: Financial resources allocated? (yes/no)

DECISION: T1 ∧ T2 ∧ T3 ∧ T4""",

"4.4.1": """TESTS:
T1: A public communication strategy (not only a technical one)? (yes/no)
T2 (alt): A development timetable, where the strategy is not yet drafted? (yes/no)
T3: Expected products (press notes, social media, multimedia)? (yes/no)

DECISION: (T1 ∧ T3)   OR   T2 with a clear timetable""",

"4.4.2": """TESTS:
T1: Human resources allocated to communication (staff/consultants)? (yes/no)
T2: Financial resources allocated (a budget line)? (yes/no)
T3: Formats accessible to persons with disabilities? (yes/no)
T4 (cond): Products in local languages / plain language / sign language where applicable? (yes/no/NA)

DECISION: T1 ∧ T2 ∧ T3 ∧ (T4 ∨ NA)""",

"4.4.3": """TESTS:
T1: Is the budget > USD 5,000,000? (yes/no/NA)
T2: Is it funded through a PPP (public-private partnership) or by the European Commission? (yes/no/NA)
T3: if T1 ∧ T2 → is the DCOMM template completed? (yes/no/NA)
T4: if T1 ∧ T2 → is there coordination with the relevant country office? (yes/no/NA)

DECISION: if T1 ∧ T2 → T3 ∧ T4; if ¬(T1 ∧ T2) → N/A""",

"5.1.1": """TESTS:
T1: Are all sections of the ILO template present? (yes/no)
T2: Is the order consistent with the guidance? (yes/no)
T3: Are the required annexes included? (yes/no)

DECISION: T1 ∧ T2 ∧ T3""",

"5.2.1": """TESTS:
T1: Are the core ideas identifiable without re-reading (a clear executive summary)? (yes/no)
T2: Absence of unnecessary jargon or evident repetition? (yes/no)
T3: Visible structure (subheadings, tables, bullets where they help)? (yes/no)

DECISION: T1 ∧ T2 ∧ T3""",
})

if __name__ == "__main__":
    import json
    import pathlib
    out = pathlib.Path(__file__).parent / "tests_si.json"
    out.write_text(json.dumps(TESTS, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(TESTS)} entradas -> {out.name}")
