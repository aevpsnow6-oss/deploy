"""English deck for the 30-min ILO training block (Ahmed).

Slide for slide the same as build_deck.py: same order, same layout, same
sober style. Layout and palette live in ilo_deck.py. Since language="en"
exists, the workbook, its headers and the reasoning are quoted as the tool
produces them for an English user. The progress messages are the exception:
they are still Spanish-only, so they are described rather than quoted.
"""

from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from ilo_deck import *  # noqa: F403 - shared layout engine
import ilo_deck

ilo_deck.set_style("sober")
prs = new_deck()

# ═══ 1 · TITLE ═════════════════════════════════════════════════════════
s = blank_slide()
rect(s, 0, 0, Inches(0.42), EMU_H, BLUE)
rect(s, Inches(0.42), 0, Inches(0.09), EMU_H, RED)
tb, tf = textbox(s, Inches(1.25), Inches(1.65), Inches(11.3), Inches(2.4))
p = tf.paragraphs[0]
r = p.add_run(); r.text = "How the technology works\nand what its limits are"
r.font.size = Pt(50); r.font.bold = True; r.font.color.rgb = BLUE
p2 = tf.add_paragraph(); p2.space_before = Pt(18)
set_runs(p2, [N("GPT Agent · PRODOC Appraisal Checklist")], 27)
rect(s, Inches(1.25), Inches(4.6), Inches(3.1), Inches(0.07), RED)
tb3, tf3 = textbox(s, Inches(1.25), Inches(4.95), Inches(11.0), Inches(1.6))
for segs in [[B("Presenter:  "), N("Ahmed Eid")],
             [B("Session:  "), N("Day 1 · Thursday 27 August 2026 · 9:30–10:30 (Lima)")],
             [B("Block:  "), N("Item 4 · 30 minutes, with questions folded in")]]:
    p = tf3.add_paragraph() if tf3.paragraphs[0].runs else tf3.paragraphs[0]
    p.space_after = Pt(8)
    set_runs(p, segs, 21)

# ═══ 2 · ROADMAP ═══════════════════════════════════════════════════════
s = slide_new("What we will cover")
steps(s, [
    ("1", "Where it comes from", "from the Streamlit app to the GPT Agent"),
    ("2", "How it is built", "the rubric, the tests, the search for evidence"),
    ("3", "Why it repeats", "randomness, 10 runs and stability"),
    ("4", "What it does NOT do", "real limits, and how it compares to the Enterprise GPT"),
    ("5", "Demonstration", "upload → run → read the Excel"),
], size=21)
band(s, [R("Questions are folded into this block: "), N("interrupt me at any point.")])

divider("Part 1", "Where it comes from and what it is", "From the Streamlit application to the GPT Agent · what a language model really is")

# ═══ 3 · FROM STREAMLIT TO GPT ═════════════════════════════════════════
s = slide_new("Where it comes from: Streamlit to the GPT Agent")
table(s, ["", "Streamlit application", "GPT Agent"],
      [[("Access", BLUE, True), "URL + environment password", "A ChatGPT link"],
       [("Install", BLUE, True), "Own server, dependencies", "None: it opens in the browser"],
       [("Learning curve", BLUE, True), "Tabs and parameters to learn", "Plain conversation"],
       [("Rubric", BLUE, True), "Loaded on the server", "Loaded on the server (same)"],
       [("Engine", BLUE, True), "The same evaluation code", "The same evaluation code"],
       [("Output", BLUE, True), "Downloadable Excel", "Downloadable Excel (same)"]],
      CL, CT, CW, [1.5, 3.2, 3.2], fsize=18, header_fs=18, fill_to=Inches(6.22))
band(s, [B("The engine did not change. "), N("What changed is the front door: the Agent removes the install and interface barrier.")])

# ═══ 3b · THE STREAMLIT APP ════════════════════════════════════════════
s = slide_new("What the Streamlit application was")
bullets(s, [
    ([B("Why it was built."), N("  The Appraisal Checklist was applied by hand, criterion by criterion. "
      "Reviewing a full PRODOC took hours of specialist time, and appraisal coverage was limited.")], 0, None),
], t=CT, h=Inches(1.05), size=19, gap=6)
table(s, ["Tab", "What it did", "Still available"],
      [[("1", BLUE, True), "Preliminary quality appraisal of PRODOCs (Appraisal Checklist)", "Yes, now also as an Agent"],
       [("2", BLUE, True), "Specific attributes diagnosis: gender, participation, just transition", "Yes, as its own Agent"],
       [("3", BLUE, True), "Project sustainability diagnosis", "Yes, as its own Agent"],
       [("4", BLUE, True), "Ask your documents: chat over one or more files", ("Streamlit only", MUTED, False)],
       [("5 and 6", BLUE, True), "Classification of evaluation recommendations (ES / EN)", ("Streamlit only", MUTED, False)]],
      CL, Inches(2.42), CW, [1.1, 5.4, 2.4], fsize=17, header_fs=17, fill_to=Inches(6.22))
band(s, [B("The first three workflows are the ones that became Agents.  "), N("Document chat and recommendation classification still live in the application.")])

# ═══ 4 · WHAT AN LLM IS ════════════════════════════════════════════════
s = slide_new("What a GPT is, in plain terms")
bullets(s, [
    ([B("A language model predicts text."), N("  It has read enormous amounts of text and learned which words tend to follow others in a given context.")], 0, BLUEMD),
    ([B("It is not a database."), N("  It does not «look up» the PRODOC the way you search a filing system: it reads the whole thing and reasons over it there and then.")], 0, BLUEMD),
    ([B("It is not a search engine."), N("  There is no list of correct answers stored somewhere that the system retrieves.")], 0, BLUEMD),
    ([B("It is a reader that follows instructions."), N("  We give it the document, the rubric and a very precise instruction, and it returns a judgement with the quotation that supports it.")], 0, BLUEMD),
], size=22, gap=20)
band(s, [R("Practical consequence:  "), N("the Agent can only see what is written in the PRODOC. What the specialist knows but did not document does not exist for it.")])

# ═══ 5 · WHAT IT DOES NOT DO ═══════════════════════════════════════════
s = slide_new("What the Agent does not do")
table(s, ["It does not do this", "Why", "What it means for you"],
      [["Approve or reject a PRODOC", "It issues no official ILO determination", "The decision stays with the appraiser"],
       ["Guess what is undocumented", "It only reads the text it is given", "A «No» may be a documentation gap, not a design gap"],
       ["Remember previous appraisals", "Every run starts from zero", "It does not compare across projects automatically"],
       ["Rewrite the PRODOC", "That is neither its purpose nor its scope", "It shows you where to look, not what to write"],
       ["Always give the same answer", "There is inherent randomness", "Which is why we repeat and measure stability"]],
      CL, CT, CW, [2.4, 2.9, 3.1], fsize=18, header_fs=18, fill_to=Inches(6.22))
band(s, [B("A deliberate design choice:  "), N("we prefer a narrow, auditable diagnosis over a broad recommendation with no grounding in the document.")])

divider("Part 2", "How it is built", "The digitised rubric, the checks and the search for evidence")

# ═══ 6 · HOW THE RUBRIC WAS BUILT IN ═══════════════════════════════════
s = slide_new("How the Appraisal Checklist was built in")
steps(s, [
    ("1", "The rubric was digitised", "each criterion became a structured row, not free text"),
    ("2", "It was broken into tests", "each criterion is expressed as boolean questions T1, T2, T3…"),
    ("3", "A decision rule was written", "an explicit formula determines Yes / Partial / No"),
    ("4", "Verifiable anchors were set", "terms and phrases the Agent must look for in the text"),
    ("5", "Subjectivity was labelled", "every criterion is tagged High / Medium / Low"),
], size=20)
band(s, [B("The rubric lives on the server. "), N("Users never upload it: every appraisal runs against the same institutional version.")])

# ═══ 7 · THE 5 SECTIONS ════════════════════════════════════════════════
s = slide_new("What it assesses: 76 criteria across 5 sections")
table(s, ["Section", "Content", "Criteria", "Subsections"],
      [[("1", BLUE, True), "Relevance", ("20", RED, True), "1.1 – 1.5"],
       [("2", BLUE, True), "Design validity", ("13", RED, True), "2.1 – 2.4"],
       [("3", BLUE, True), "Results framework and M&E", ("27", RED, True), "3.1 – 3.7"],
       [("4", BLUE, True), "Implementation", ("14", RED, True), "4.1 – 4.4"],
       [("5", BLUE, True), "Presentation", ("2", RED, True), "5.1 – 5.2"]],
      CL, CT, CW, [1.1, 4.0, 1.2, 1.9], fsize=20, header_fs=19, fill_to=Inches(6.22))
band(s, [B("You can filter:  "), N("the full rubric, one section («only section 3»), subsections («only 1.1 and 2.3») or specific criteria («only 1.1.1 and 3.1.1»).")])

# ═══ 8 · CRITERION TO TESTS ════════════════════════════════════════════
s = slide_new("From criterion to tests: how it «reads» a criterion")
rect(s, CL, CT, Inches(0.05), Inches(1.30), BLUEMD)
tbc, tfc = textbox(s, CL + Inches(0.22), CT, CW - Inches(0.3), Inches(1.30))
pc = tfc.paragraphs[0]
set_runs(pc, [B("Criterion 1.5.6"), N("   ·   Section 1 Relevance   ·   subjectivity "), R("High")], 15)
pc2 = tfc.add_paragraph(); pc2.space_before = Pt(4)
set_runs(pc2, [I("«Wherever possible, the proposal promotes and highlights the use of a "
                 "gender-transformative approach.»")], 19)
pc3 = tfc.add_paragraph(); pc3.space_before = Pt(6)
set_runs(pc3, [N("Written like this it cannot be assessed: it is broken into closed questions.")], 15)
table(s, ["Test", "Closed question", "Answer"],
      [[("T1", BLUE, True), "Does it distinguish the type of approach (sensitive / responsive / transformative)?", ("yes / no", GRAY, True)],
       [("T2", BLUE, True), "Does it articulate how the project challenges norms or power relations?", ("yes / no", GRAY, True)],
       [("T3", BLUE, True), "Are there dedicated actions to transform relations, not just to include women?", ("yes / no", GRAY, True)]],
      CL, Inches(2.78), CW, [0.75, 6.2, 1.15], fsize=20, header_fs=19, fill_to=Inches(4.95))
rect(s, CL, Inches(5.20), CW, Inches(0.78), TINT)
rect(s, CL, Inches(5.20), Inches(0.05), Inches(0.78), BLUEMD)
tb, tf = textbox(s, CL + Inches(0.22), Inches(5.20), CW - Inches(0.3), Inches(0.78), MSO_ANCHOR.MIDDLE)
p = tf.paragraphs[0]
set_runs(p, [B("DECISION:   "), N("T1 ∧ T2 ∧ T3 → "), ("Yes", RED, True), N("   ·   T1 ∨ T3, not all three → "), ("Partial", RED, True), N("   ·   ¬T1 ∧ ¬T3 → "), ("No", RED, True)], 17)
band(s, [B("This is what makes the result auditable:  "), N("you can check, test by test, where and why the Agent reached that conclusion.")])

# ═══ 9 · HOW IT LOOKS FOR EVIDENCE ═════════════════════════════════════
s = slide_new("How it looks for evidence: DEDICATED vs FRAMING")
bullets(s, [
    ([B("A word appearing in the document is not enough."), N("  The Agent classifies each mention before counting it as evidence.")], 0, None),
], t=CT, h=Inches(0.9), size=21, gap=6)
table(s, ["", "FRAMING  (does not count)", "DEDICATED  (does count)"],
      [[("What it is", BLUE, True), "The topic is mentioned, with no development of its own", "The topic has its own space in the design"],
       [("Examples", BLUE, True), "Lists of ≥3 groups · generic inclusion language · «among others» enumerations", "An output or result naming it · a disaggregated indicator · a dedicated activity · a budget line · a quantified target"]],
      CL, Inches(2.15), CW, [1.3, 3.3, 3.6], fsize=18, header_fs=18, fill_to=Inches(6.22))
band(s, [R("Hard rule:  "), N("if all citable evidence is framing, the result must be «No» or «Not Found», however many times the topic is named.")])

# ═══ 9b · A FULL WORKED CASE ═══════════════════════════════════════════
s = slide_new("A full case: from PRODOC text to verdict")

LX, LW = CL, Inches(5.85)
RX, RW = CL + LW + Inches(0.38), CW - LW - Inches(0.38)
TY, PH = Inches(1.26), Inches(4.72)

# ── left · what the document says ─────────────────────────────────────
rect(s, LX, TY, LW, Inches(0.42), BLUE)
tbh, tfh = textbox(s, LX + Inches(0.14), TY, LW - Inches(0.2), Inches(0.42), MSO_ANCHOR.MIDDLE)
set_runs(tfh.paragraphs[0], [WB("WHAT THE PRODOC SAYS")], 15)
rect(s, LX, TY + Inches(0.42), LW, PH - Inches(0.42), GRAYLT)
tbd, tfd = textbox(s, LX + Inches(0.18), TY + Inches(0.54), LW - Inches(0.36), PH - Inches(0.66))
EXTRACTS = [
    [B("§ 2.3 Gender approach")],
    [I("«The project adopts a gender-transformative approach, distinguishing it from merely sensitive or responsive approaches.»")],
    [N("")],
    [B("§ 3.1 Beneficiaries")],
    [I("«Priority will be given to the participation of women, young people, persons with disabilities and indigenous peoples.»")],
    [N("")],
    [B("§ 4.2 Activity 2.4")],
    [I("«Working sessions with cooperatives to review the credit-access criteria that exclude women titleholders. Budget: USD 18,000.»")],
]
first = True
for segs in EXTRACTS:
    par = tfd.paragraphs[0] if first else tfd.add_paragraph()
    first = False
    par.space_after = Pt(3)
    set_runs(par, segs, 14)

# ── right · how the Agent assesses it ─────────────────────────────────
rect(s, RX, TY, RW, Inches(0.42), BLUE)
tbh2, tfh2 = textbox(s, RX + Inches(0.14), TY, RW - Inches(0.2), Inches(0.42), MSO_ANCHOR.MIDDLE)
set_runs(tfh2.paragraphs[0], [WB("HOW THE AGENT ASSESSES IT")], 15)

CHECKS = [
    ("✓", BLUEMD, "T1 · Distinguishes the type of approach", "§ 2.3 names and distinguishes the three approaches."),
    ("✗", RED, "T2 · Articulates how it challenges power norms", "it declares the approach but never explains the mechanism."),
    ("✓", BLUEMD, "T3 · Dedicated actions to transform relations", "§ 4.2 revises credit-access rules, with its own budget."),
    ("–", GRAY, "§ 3.1 does not count as evidence", "a list of four groups with no follow-through: FRAMING, not DEDICATED."),
]
top = TY + Inches(0.52)
alto = Inches(1.02)
grupo = []
for marca, color, titulo, detalle in CHECKS:
    rect(s, RX, top, RW, alto, GRAYLT if marca != "–" else WHITE)
    rect(s, RX, top, Inches(0.05), alto, color)
    tbm, tfm = textbox(s, RX + Inches(0.12), top, Inches(0.5), alto, MSO_ANCHOR.MIDDLE)
    pm = tfm.paragraphs[0]; pm.alignment = PP_ALIGN.CENTER
    set_runs(pm, [(marca, color, True)], 20)
    tbc, tfc = textbox(s, RX + Inches(0.62), top, RW - Inches(0.78), alto, MSO_ANCHOR.MIDDLE)
    pc = tfc.paragraphs[0]
    set_runs(pc, [B(titulo)], 13.5)
    pc2 = tfc.add_paragraph(); pc2.space_before = Pt(1)
    set_runs(pc2, [N(detalle)], 12.5)
    grupo.append(tbc)
    top = top + alto + Inches(0.11)
register_group(grupo)

band(s, [B("DECISION:  "), N("T1 ∧ T2 ∧ T3 required. T1 and T3 met, T2 missing  →  "), R("PARTIAL"), N("   ·   illustrative extract, not a real PRODOC.")])

divider("Part 3", "Why it repeats", "Randomness, ten runs and what stability means")

# ═══ 10 · WHY IT REPEATS ═══════════════════════════════════════════════
s = slide_new("Why every criterion is run 10 times")
bullets(s, [
    ([B("A language model is not deterministic."), N("  The same question about the same document can produce different answers.")], 0, BLUEMD),
    ([B("Asking once would be fragile."), N("  We would be handing you the result of a single roll.")], 0, BLUEMD),
    ([B("The fix: repeat and consolidate."), N("  Each criterion is evaluated 10 times independently and the most frequent result (the mode) is taken.")], 0, BLUEMD),
    ([B("Disagreement is information, not noise."), N("  How many of the 10 runs agreed is reported to you, and you should read it.")], 0, BLUEMD),
], size=22, gap=20)
band(s, [B("A full appraisal is "), N("76 criteria × 10 runs ≈ "), R("760 model calls"), N(". That is why it takes minutes, and why filtering by section or criterion pays off.")])

# ═══ 11 · WHERE RANDOMNESS COMES FROM ══════════════════════════════════
s = slide_new("Where the randomness comes from")
table(s, ["Source", "What happens", "How we control it"],
      [["Probabilistic generation", "The model picks among possible continuations; not always the same one", "10 runs + modal result"],
       ["Ambiguous criteria", "Where the rubric admits a reading, the model may lean differently", "Closed tests and explicit decision rules"],
       ["Scattered evidence", "If evidence is spread out, different runs quote different passages", "The DEDICATED / FRAMING filter"],
       ["Long documents", "More text, more room for partial reading", "Structured extraction, one criterion at a time"]],
      CL, CT, CW, [2.3, 3.5, 2.6], fsize=18, header_fs=18, fill_to=Inches(6.22))
band(s, [B("Randomness is not eliminated: it is measured and reported. "), N("That is better than hiding it behind a single answer.")])

# ═══ 12 · STABILITY ════════════════════════════════════════════════════
s = slide_new("What «stability» means")
tb, tf = textbox(s, CL, CT, CW, Inches(1.5))
p = tf.paragraphs[0]
set_runs(p, [B("Stability (%)  =  "), N("of the 10 runs, how many agreed with the final result.")], 26)
p2 = tf.add_paragraph(); p2.space_before = Pt(10)
set_runs(p2, [N("Institutional threshold: "), R("80%"), N(". Below it, the criterion is flagged for human review.")], 22)
table(s, ["Stability", "Reading", "What to do"],
      [[("100 – 80%", BLUE, True), "The model was consistent", "Treat as a solid diagnosis; still verify the evidence"],
       [("79 – 50%", RED, True), "The runs disagreed", "Mandatory human review; read the «Alternative result»"],
       [("< 50%", RED, True), "No real consensus", "The criterion is ambiguous or the evidence thin: you decide"]],
      CL, Inches(3.05), CW, [1.6, 2.8, 4.0], fsize=19, header_fs=18, fill_to=Inches(6.22))
band(s, [B("Stability does not measure whether the Agent was right. "), N("It measures how far the runs agreed. A unanimous 100% can still be wrong: in 1.1.1, ten out of ten said «Yes», reading «aligned with» as contribution.")])

# ═══ 13 · SUBJECTIVITY ═════════════════════════════════════════════════
s = slide_new("Subjective criteria and human review")
bullets(s, [
    ([B("Every criterion carries a subjectivity label: "), R("High · Medium · Low"), N(".")], 0, BLUEMD),
    ([B("High subjectivity"), N(" means the judgement depends on institutional context, not just the text. Example: whether a gender approach is «transformative».")], 0, BLUEMD),
    ([B("They get more reasoning"), N(" from the model and are flagged in the «Human review recommended» column.")], 0, BLUEMD),
    ([B("The «Priority Review» sheet is something else: "), N("what must be reviewed, full stop. Unstable criteria (<80%) and «No» verdicts; about 17–20 on a full PRODOC.")], 0, BLUEMD),
    ([B("High subjectivity does not enter that sheet:"), N("  you review it at your discretion.")], 0, BLUEMD),
], size=21, gap=15)
band(s, [R("The order of work:  "), N("first «Priority Review», which is mandatory; then high subjectivity, where your judgement adds the most.")])

divider("Part 4", "Limits and comparison", "What it costs, what it does not do, and how it differs from the Enterprise GPT")

# ═══ 14 · TRADE-OFFS ═══════════════════════════════════════════════════
s = slide_new("Strengths and trade-offs of this technology")
table(s, ["", "Strength", "Trade-off / cost"],
      [[("Coverage", BLUE, True), "Reviews all 76 criteria without fatigue", "It reviews what is written, not what is known"],
       [("Time", BLUE, True), "Frees specialist hours to work on the gaps", "A full run takes several minutes"],
       [("Traceability", BLUE, True), "Quotes the evidence and shows the reasoning per test", "It produces a lot of detail: you must know how to read it"],
       [("Consistency", BLUE, True), "Applies the same rubric to every project", "Not deterministic: hence the repeats"],
       [("Cost", BLUE, True), "Low per document against the equivalent human time", "Billed per call: the budget is finite"]],
      CL, CT, CW, [1.5, 3.5, 3.3], fsize=18, header_fs=18, fill_to=Inches(6.22))
band(s, [B("Tokens and budget:  "), N("cost depends on the size of the PRODOC and how many criteria are run. Full rubric: ~760 calls; five specific criteria: 50.")])

# ═══ 15 · ENTERPRISE GPT ═══════════════════════════════════════════════
s = slide_new("What about the ILO Enterprise GPT?")
table(s, ["", "Enterprise GPT", "Appraisal Checklist Agent"],
      [[("Rubric", BLUE, True), "Would have to be uploaded in every conversation", "Server-side, single version"],
       [("Application", BLUE, True), "General reading of the attached file", "Each criterion run separately, with its tests"],
       [("Repetition", BLUE, True), "One single pass", "10 runs per criterion + modal result"],
       [("Stability", BLUE, True), "Not available", "Reported per criterion, 80% threshold"],
       [("Output", BLUE, True), "Text in the conversation", "Structured Excel, auditable and archivable"],
       [("Access today", BLUE, True), ("No access to the Agent from the enterprise environment", RED, True), "A ChatGPT link"]],
      CL, CT, CW, [1.5, 3.2, 3.6], fsize=17, header_fs=17, fill_to=Inches(6.22))
band(s, [B("The added value is not «using AI»:  "), N("it is the institutional rubric applied criterion by criterion, repeated and measured. A generic GPT does not reproduce that.")])

# ═══ 16 · THREE LIMITS ═════════════════════════════════════════════════
s = slide_new("The three limits to keep in mind")
steps(s, [
    ("1", "It is a complete first review,", "not a deterministic diagnosis. It flags gaps for you to examine."),
    ("2", "It works on the document's evidence,", "not on the real project or on what you know about it."),
    ("3", "It does not replace the appraiser's judgement.", "The result is an input; the determination stays human."),
], t=Inches(1.9), size=21)
band(s, [R("No output constitutes an official ILO determination. "), N("It is an AI-assisted assessment that requires expert validation.")])

divider("Part 5", "The tool in use", "Upload · scope · run · download · interpret")

# ═══ 18 · THE FLOW ═════════════════════════════════════════════════════
s = slide_new("The full flow, in five steps")
steps(s, [
    ("1", "Upload the PRODOC", "one .docx file per appraisal"),
    ("2", "Set the scope", "full rubric, sections, subsections or specific criteria"),
    ("3", "Run", "the Agent starts the job; you type «status» to see progress"),
    ("4", "Download the Excel", "it is the auditable record of the appraisal"),
    ("5", "Interpret", "locate the evidence, starting with the Priority Review sheet"),
], size=21)
band(s, [B("State the scope in the same message as the upload:  "), N("it saves a round of questions and caps the cost from the start.")])

# ═══ 19 · STEPS 1-2 ════════════════════════════════════════════════════
s = slide_new("Steps 1 and 2 · Upload and scope")
bullets(s, [
    ([B("Open the Agent and say hello."), N("  It introduces itself: what it assesses, which sections exist, what you can filter. That costs no appraisal.")], 0, BLUEMD),
    ([B("Attach a single .docx."), N("  If you upload several, the Agent will ask which one: one document per appraisal.")], 0, BLUEMD),
    ([B("Say what you want assessed."), N("  The report comes in the language you write in. Examples:")], 0, BLUEMD),
    ([I("«Assess this PRODOC against the full rubric»")], 1, None),
    ([I("«Assess only section 3 (results framework) and summarise the gaps»")], 1, None),
    ([I("«Assess only criteria 1.1.1, 2.1.2 and 3.1.1»")], 1, None),
], size=21, gap=13)
band(s, [R("Where to start:  "), N("one section first. It arrives sooner, costs less, and lets you calibrate your reading before a full run.")])

# ═══ 20 · STEP 3 ═══════════════════════════════════════════════════════
s = slide_new("Step 3 · What happens while it runs")
steps(s, [
    ("1", "The Agent downloads the document", "and extracts its text by section"),
    ("2", "It loads the rubric and filters your scope", "and confirms the scope back to you: check it matches"),
    ("3", "It runs the evaluations in parallel", "each criterion, 10 times, up to 48 concurrent calls"),
    ("4", "It consolidates each criterion", "modal result + stability percentage"),
    ("5", "It builds the Excel", "and attaches it to the conversation"),
], size=20)
band(s, [B("Type «status» to check progress:  "), N("the Agent replies with how many model calls are done and the time left. It does not report on its own.")])

# ═══ 21 · THE EXCEL ════════════════════════════════════════════════════
s = slide_new("Step 4 · The results Excel")
bullets(s, [
    ([B("1 · «Appraisal Result»"), N("  — one row per criterion. The auditable record: keep it.")], 0, None),
    ([B("2 · «Priority Review»"), N("  — unstable and «No», with the reason. "), R("Mandatory review.")], 0, None),
    ([B("3 · «Rubric Applied»"), N("  — the checks and the rule for each criterion assessed.")], 0, None),
], t=CT, h=Inches(1.55), size=19, gap=4)
table(s, ["Column group", "What it holds", "What it is for"],
      [[("Identification", BLUE, True), "ID · Subsection · Criterion · Cross-cutting", "Locate the criterion in the Checklist"],
       [("Result", BLUE, True), "Verdict (Yes / Partial / No / Not Found / N/A)", "The diagnosis for that criterion"],
       [("Confidence", BLUE, True), "Stability (%) · Stable (≥80%) · Alternative result", "How far the 10 runs agreed"],
       [("Support", BLUE, True), "Reasoning (check by check) · Evidence", "Verify why it reached that result"],
       [("Priority", BLUE, True), ("Human review recommended", RED, True), "Also flags high subjectivity"]],
      CL, Inches(2.82), CW, [2.0, 3.7, 2.6], fsize=16, header_fs=16, fill_to=Inches(6.22))
band(s, [B("Start with sheet 2.  "), N("Sheet 3 makes the file self-contained: an auditor sees the rule next to the verdict. In Spanish: «Resultado Diagnostico», «Revisión prioritaria», «Rubrica aplicada».")])

# ═══ 22 · READING A ROW ════════════════════════════════════════════════
s = slide_new("How to read a row, in order")
steps(s, [
    ("1", "Look at the Verdict", "Yes / Partial / No / Not Found / N/A"),
    ("2", "Look at the Stability", "did the 10 runs agree, or was there disagreement?"),
    ("3", "Read the Evidence", "does that passage genuinely support the result?"),
    ("4", "Read the Reasoning", "which check failed, and on what grounds"),
    ("5", "Decide", "design gap, documentation gap, or Agent error?"),
], size=21)
band(s, [R("«Not Found» is not «No»:  "), N("«No» says the criterion is not met; «Not Found» says the document does not let you tell. And never accept an answer without opening the evidence.")])

# ═══ 22b · ANATOMY OF THE REASONING ════════════════════════════════════
s = slide_new("What the Reasoning column contains")

# block reproducing the cell exactly as it appears in the English Excel
BX, BY, BW = CL, Inches(1.30), Inches(8.15)
rect(s, BX, BY, BW, Inches(4.92), GRAYLT)
rect(s, BX, BY, Inches(0.07), Inches(4.92), BLUEMD)
tbb, tfb = textbox(s, BX + Inches(0.22), BY + Inches(0.12), BW - Inches(0.42), Inches(4.7))
CELL = [
    [B("WHY PARTIAL"), N(" · 2 of 3 checks are met.")],
    [N("Missing: articulate how the project challenges power norms.")],
    [N("")],
    [B("VERIFICATION")],
    [("✓ ", BLUEMD, True), N("Does it distinguish the type of approach?")],
    [I("      section 3.2 distinguishes the three approaches.")],
    [("✗ ", RED, True), N("Does it articulate how it challenges power norms?")],
    [I("      the mechanism is not explained in any section.")],
    [("✓ ", BLUEMD, True), N("Dedicated actions to transform relations?")],
    [I("      activity 2.4, with an allocated budget.")],
    [N("")],
    [B("STABILITY"), N(" · 6 of 10 runs agreed. Alternative result: No.")],
    [B("RULE"), N(" · all 3 checks are required  (T1 ∧ T2 ∧ T3)")],
]
first = True
for segs in CELL:
    par = tfb.paragraphs[0] if first else tfb.add_paragraph()
    first = False
    par.space_after = Pt(3)
    set_runs(par, segs, 15)

# the four parts, read on the right
RX = CL + BW + Inches(0.30)
RW = CW - BW - Inches(0.30)
parts = [
    ("WHY", "the reason, in one line. If you are in a hurry, read only this."),
    ("VERIFICATION", "each check stated in full, with ✓ / ✗ and its justification."),
    ("STABILITY", "how many of the 10 runs agreed."),
    ("RULE", "the formal Checklist rule, for audit."),
]
top = BY
grupo_partes = []
for titulo, detalle in parts:
    rect(s, RX, top, RW, Inches(1.16), WHITE)
    rect(s, RX, top, Inches(0.055), Inches(1.16), RED)
    tbp, tfp = textbox(s, RX + Inches(0.18), top, RW - Inches(0.3), Inches(1.16), MSO_ANCHOR.MIDDLE)
    pp = tfp.paragraphs[0]
    set_runs(pp, [B(titulo)], 17)
    pp2 = tfp.add_paragraph(); pp2.space_before = Pt(2)
    set_runs(pp2, [N(detalle)], 14)
    grupo_partes.append(tbp)
    top = top + Inches(1.25)
register_group(grupo_partes)

band(s, [B("No more decoding «T1 ∧ T2 ∧ T3»:  "), N("each check reads on its own. The formal rule stays at the foot, for anyone tracing the result.")])

# ═══ 23 · LOCATING EVIDENCE ════════════════════════════════════════════
s = slide_new("Step 5 · Locating the evidence in the PRODOC")
bullets(s, [
    ([B("The Evidence column holds the verbatim quotation"), N(" the Agent used. Search for it in the PRODOC with Ctrl+F to see it in context.")], 0, BLUEMD),
    ([B("When the evidence is an absence,"), N(" the Agent says so explicitly: «No section X was found». That is a verifiable finding too.")], 0, BLUEMD),
    ([B("Three possible outcomes when you verify:")], 0, BLUEMD),
    ([R("The information is there and the Agent missed it "), N("→ discard the automated diagnosis")], 1, None),
    ([R("It is not written, but the formulator knows it "), N("→ improve the PRODOC")], 1, None),
    ([R("It does not exist yet "), N("→ resolve before closing formulation, or carry into inception")], 1, None),
], size=21, gap=13)
band(s, [B("The formula:  "), N("does it exist?  →  is it sufficient?  →  is it documented?")])

# ═══ 24 · LINKING ══════════════════════════════════════════════════════
s = slide_new("Linking to other resources")
table(s, ["Resource", "What it adds", "Example use"],
      [[("Chat with the Agent", BLUE, True), "Ask about the result already produced, in plain language", "«Summarise the section 3 gaps in three points»"],
       [("The other GPTs built", BLUE, True), "Specific Attributes (gender, participation, just transition) and Sustainability", "Go deeper on a theme the Checklist only touches transversally"],
       [("i-EVal chatbot", BLUE, True), "Evidence from previous evaluations in the region", "«What recurring problems have evaluations found in projects with labour ministries?»"]],
      CL, CT, CW, [2.2, 3.4, 4.0], fsize=18, header_fs=18, fill_to=Inches(6.22))
band(s, [B("The Excel is not the end of the process:  "), N("it is the starting point of the conversation with the formulating team.")])

# ═══ 25 · FAQ ══════════════════════════════════════════════════════════
s = slide_new("Frequently asked questions")
table(s, ["Question", "Short answer"],
      [["Why can an answer change?", "The model is not deterministic. Hence 10 repeats and a reported stability figure."],
       ["What does 50–70% stability mean?", "The runs disagreed: ambiguous criterion or thin evidence. Human review."],
       ["Why did it miss something I know is there?", "It only reads the document. If it is there and was missed, verify and discard the diagnosis."],
       ["Why no improvement recommendations?", "It holds no knowledge beyond the document. Recommendations come from your reading."],
       ["What does a PRODOC cost to assess?", "It depends on size and scope: full rubric, ~760 model calls; five specific criteria, 50."],
       ["Can I assess other documents?", "Dedicated agents exist for gender, participation, just transition and sustainability."]],
      CL, CT, CW, [3.4, 6.0], fsize=17, header_fs=17, fill_to=Inches(7.05))

divider("Closing", "What to remember", "")

# ═══ 26 · CLOSING ══════════════════════════════════════════════════════
s = slide_new("What to take away from this block")
bullets(s, [
    ([B("The Agent reads the document, not the project."), N("  A «No» may be a design gap or simply something undocumented. Telling them apart is your job.")], 0, RED),
    ([B("It runs 10 times and tells you how far they agreed."), N("  Stability is a measure of internal confidence, not of correctness.")], 0, RED),
    ([B("Every result comes with its evidence."), N("  Always verify it: that is what makes the diagnosis defensible to the formulating team.")], 0, RED),
    ([B("The «Priority Review» sheet is your starting point."), N("  Unstable and «No»: mandatory review.")], 0, RED),
], size=22, gap=20)
band(s, [B("The value of the tool depends on the quality of the human review that follows."), N("")], bg=BLUELT)

# ═══ 27 · QUESTIONS ════════════════════════════════════════════════════
s = blank_slide()
rect(s, 0, 0, EMU_W, EMU_H, BLUE)
rect(s, Inches(1.2), Inches(3.05), Inches(3.4), Inches(0.09), RED)
tb, tf = textbox(s, Inches(1.2), Inches(3.3), Inches(11.0), Inches(1.6))
p = tf.paragraphs[0]
r = p.add_run(); r.text = "Questions"
r.font.size = Pt(54); r.font.bold = True; r.font.color.rgb = WHITE
p2 = tf.add_paragraph(); p2.space_before = Pt(14)
set_runs(p2, [W("GPT Agent · PRODOC Appraisal Checklist   ·   Ahmed Eid")], 22)

ilo_deck.finalize("ILO_Appraisal_GPT_Training_Ahmed_EN.pptx")
