# Script · Block 4 · "How the technology works and what its limits are"
**Ahmed Eid · Day 1, Thursday 27 Aug 2026 · 9:30–10:30 (Lima) · 30 min, questions folded in**

> **Time split.** 30 min of presentation + the 13 min of questions the agenda folds into this
> block = **43 real minutes**. The script below sums to **30:30**: running slightly over is
> fine and expected. The ✂ marks are there in case an earlier block overruns.

| Part | Slides | Time | Cumulative |
|---|---|---|---|
| Opening | 1–2 | 1:30 | 1:30 |
| A · Where it comes from, what it is | 3–5 | 4:00 | 5:30 |
| B · How it is built | 6–10 | 7:15 | 12:45 |
| C · Why it repeats | 11–14 | 5:30 | 18:15 |
| D · Limits and comparison | 15–17 | 3:30 | 21:45 |
| E · Demonstration | 18–25 | 7:45 | 29:30 |
| Close | 26–29 | 1:00 | 30:30 |

---

## Opening · slides 1–2 · 1:30

**[Sl. 1]**
"Good morning. The previous blocks covered *what* the Agent does and *who* should use it.
My part is under the bonnet: **how it works, why it sometimes changes its mind, and what it
cannot do**. At the end we will see it running."

**[Sl. 2]** Read the five points aloud, briskly.
"One warning: questions are folded into my block. **Don't save them for the end** — if
something is unclear, interrupt me, because it is probably unclear to someone else too."

---

## Part A · Where it comes from, what it is · slides 3–5 · 4:00

**[Sl. 3 · Streamlit to the Agent] — 1:30**
"This did not start as a GPT. It started as a Streamlit application that already existed and
some of you have seen. It worked, but it had two barriers: you had to install it, and you had
to learn its interface."

Point at the last row: "**The engine is the same code.** What changed is the front door. That
matters, because this is not a brand-new untested tool: it is the same evaluation logic,
served differently."

**[Sl. 4 · What a GPT is] — 1:45**
"Let me take the mystery out of it. A language model **predicts text**. It has read vast
amounts of text and learned which words tend to follow others."

Go point by point. Stop on the third:
"It is not a search engine. There is no list of correct answers stored somewhere that the
system retrieves. Every time it evaluates, it **reasons there and then** about the document
you gave it."

Close on the red band — **this is the most important sentence in my block**:
"The Agent only sees what is written in the PRODOC. What the specialist knows but did not
document **does not exist** for it. We will come back to this several times."

**[Sl. 5 · What it does not do] — 0:45**
Read only the left column, quickly. "Five things it does not do. One I will dwell on: it does
not always give the same answer. That sounds like a defect, and in a minute I will explain why
it is unavoidable and what we did about it."

---

## Part B · How it is built · slides 6–9 · 6:00

**[Sl. 6 · How the rubric was built in] — 1:30**
"The Appraisal Checklist was not 'attached' to the model as a file. It was **digitised
criterion by criterion**."

Walk the five steps. Emphasise step 3:
"Every criterion has an **explicit, written decision rule**. It is not the model deciding
freely what deserves a Yes."

**[Sl. 7 · The five sections] — 1:00**
"76 criteria, 5 sections. Section 3, the results framework, is the heaviest at 27."

Band: "**You can filter.** And I recommend you do. Assessing only section 3 takes less time,
costs less, and is easier to review than 76 criteria at once."

**[Sl. 8 · From criterion to tests] — 2:30** ← *the conceptual core of the block*
"This is the heart of how it works. Take a real criterion: 1.5.6, transformative approach to
gender."

"The Agent does **not** ask itself 'does this PRODOC have a good gender approach?' — that
would be an impression. It asks three closed questions, one at a time." — read T1, T2, T3.

"Then it applies a formula." — point to the DECISION line.
"All three met: Yes. Only some: Partial. Neither anchor: No."

Close: "That is what makes the result **auditable**. When you see 'Partial' on this criterion,
you can open the Excel and see exactly which check failed, and on what quotation."

**[Sl. 9 · DEDICATED vs FRAMING] — 1:00**
"A classic problem: a PRODOC mentions 'gender' fourteen times and does nothing about it."

"The Agent classifies each mention before counting it. If the topic appears in a list of
groups, or in generic inclusion language, that is **framing** and does not count as evidence.
It counts if there is an output, a disaggregated indicator, an activity or a budget line."

Red band: "If all the evidence is framing, the result **must** be No, however many times the
word appears."

**[Sl. 10 · A full worked case] — 1:15** ← *the slide that anchors everything before it*
"So far I have explained the mechanism. Let's watch it run against real text."

Left: "Three extracts from a PRODOC. Illustrative — not a real document."

Right, one at a time: "The first check passes: section 2.3 names and distinguishes the three
approaches. The second fails: it declares the approach but never explains the mechanism. The
third passes: activity 2.4 revises credit-access rules, and it has its own budget."

Point at the fourth card — **this is the most contested point**:
"Look at section 3.1. It explicitly names four groups, women among them. And it still **does
not count as evidence**, because it is a list with no follow-through. That is framing, not
dedicated. This is exactly where people say «but we did mention it»."

Close: "All three are required. Two are met. Result: Partial. And the whole path from text to
verdict is verifiable."

---

## Part C · Why it repeats · slides 10–13 · 5:30

**[Sl. 11 · Why 10 times] — 1:30**
"Here is the answer to 'why can an answer change?'."

"A language model is **not deterministic**. The same question about the same document can give
different answers. That is not a programming bug: it is how it works."

"Asking once would hand you the result of **a single roll**. So every criterion is assessed
**10 times** independently and the most frequent result is taken."

Band: "76 criteria times 10 runs is roughly **760 model calls**. That is why it takes minutes
rather than seconds, and why I keep pushing you to filter."

**[Sl. 12 · Where randomness comes from] — 1:00** ✂ *(cuttable to 30 s: read only the first row
and the band)*
"Four sources. The first is inherent to the model; the other three we mitigate by design."

**[Sl. 13 · Stability] — 2:00** ← *the concept they will use most*
"Of those 10 runs, how many agreed? That percentage is the **stability**, and it is in the
Excel, criterion by criterion."

Walk the table. Stop on the middle red row:
"If you see 60%, that means six of ten runs said one thing and four said another. That is
**not** a result you can use as it stands. It is a signal that this criterion needs your
judgement."

Close on the band — **say this slowly**:
"Stability does **not** measure whether the Agent was right. It measures how confident it was
internally. A confidently wrong 100% is entirely possible. That is why you always, always
verify the evidence."

**[Sl. 14 · Subjectivity] — 1:00**
"Besides stability, every criterion carries a subjectivity label."

"High subjectivity means the judgement depends on institutional context, not just on the text.
Whether a gender approach is 'transformative' is arguable between two experts; whether the
PRODOC has a logframe is verifiable."

"Two things trigger the human-review flag: **high subjectivity**, or **stability below 80%**.
That column is your work queue."

---

## Part D · Limits and comparison · slides 14–16 · 3:30

**[Sl. 15 · Strengths and trade-offs] — 1:15**
Read in pairs: "Coverage: it reviews all 76 without tiring — but it reviews what is written,
not what is known."

Band: "Cost depends on document size and how many criteria you run. **Filtering by section is
the direct saving lever.**"

**[Sl. 16 · Enterprise GPT] — 1:15** ← *guaranteed question from the room*
"Someone will ask: 'what if I use the ILO Enterprise GPT and upload the rubric?'"

Walk the table quickly. Stop on two rows: repetition and stability.
"The Enterprise GPT makes **one single pass** and tells you nothing about its own consistency."

Red row: "And today there is no access to this Agent from the enterprise environment."

Band: "The added value is not 'using AI'. It is the institutional rubric applied criterion by
criterion, repeated and measured."

**[Sl. 17 · The three limits] — 1:00**
Read the three plainly. Close: "No output is an official ILO determination."

---

## Part E · Demonstration · slides 17–24 · 7:45

> **⚠ Mandatory preparation.** Have the Excel from an earlier run **already downloaded and
> open**. If you run live, launch **a single subsection** and keep talking while it runs. Never
> leave the room watching a progress bar.
>
> **⚠ Language note.** The Agent produces its output **in Spanish** — column headers,
> reasoning and evidence. Say so up front in an English session so nobody is surprised:
> *"the interface is English, the analysis comes back in Spanish."*

**[Sl. 18 · divider] — 0:15** "Let's look at it."

**[Sl. 19 · The flow] — 0:45** Walk the five steps.
Band: "State the scope in the same message as the upload."

**[Sl. 20 · Steps 1–2] — 1:00**
"If you open the Agent and say 'hello', it introduces itself and tells you what you can filter.
That costs no appraisal." — read the three example instructions.

**[Sl. 21 · Step 3] — 0:45**
"While you wait, this is what is happening." — walk the steps.
Band: "You no longer have to ask whether it finished: the Agent reports progress on its own —
'180 of 760, 24%, about 3 minutes left' — until it delivers the file. The first run of the day
may be slower to start, because the service suspends when idle."

**[Sl. 22 · The Excel] — 1:30**
"Two sheets. The first, «Resultado Diagnostico»: one row per criterion, five column
groups." — walk them. "The second, «Rubrica aplicada»: the definition of each criterion
assessed, so the file can be audited on its own."
"The headers are in Spanish, because that is what the file produces."
Band: "**'Not Found' is not 'No'.** 'No' says the criterion is not met. 'Not Found' says the
document does not let you tell. What follows is different: one is redesign, the other is
documentation."

**[Sl. 23 · Reading a row] — 1:15**
"In this order, every time." — walk the five steps.
Band: "Never accept an answer without opening the evidence."

**[Sl. 24 · What the Razonamiento contains] — 1:15** ← *the slide that prevents the most questions*
"This is the cell that used to be intimidating, and we have rewritten it completely."

Walk the four parts, pointing at the block on the left:
"At the top, **POR QUÉ** — the reason in one line. If you are in a hurry, read only that:
'2 of 3 checks met; missing: articulating how it challenges power norms'."

"Below, **VERIFICACIÓN** — each check written out in full, with its mark and its justification.
Notice it no longer says 'T2: false'. It states the whole question. You don't have to go
hunting for what T2 was."

"Then **ESTABILIDAD**, which you know, and finally **REGLA** — the formal Checklist rule, for
anyone who needs to trace the result back to the rubric."

Close: "The analysis is exactly the same as before: same checks, same evidence, same rule. All
that changed is that you can now read it without decoding anything."

**[Sl. 25 · Locating evidence] — 1:00**
"The quotation is in the Excel. Copy it and find it in the PRODOC with Ctrl+F."

The three outcomes — **this is the bridge to day 2**:
"If the information was there: discard the diagnosis. If the formulator knows it but it is not
written: improve the PRODOC. If it does not exist: resolve it before closing formulation."

Band: "Does it exist? Is it sufficient? Is it documented?"

---

## Close · slides 25–28 · 1:00

**[Sl. 26 · Linking] — 0:30** ✂ *(cuttable if time is tight)*
"The Excel is not the end. You can keep talking to the Agent about the result, and there are
dedicated agents for gender, participation, just transition and sustainability."

**[Sl. 28 · What to take away] — 0:30**
"Four things." — read them.
Close: "The value of the tool depends on the quality of the human review that follows. It saves
you the mechanical reading so you can spend your time where it counts: deciding."

**[Sl. 29]** "Questions."

---

## Pocket answers (13 min of questions)

**Why can an answer change?**
The model is not deterministic. That is why we run each criterion 10 times and report how far
they agreed. If variability worries you, stability is precisely the metric that makes it
visible.

**What does 50–70% stability mean?**
The runs did not agree. It almost always points to an ambiguous criterion or thin evidence in
the document. It is an invitation for you to look, not a result to use as it stands.

**Why did it miss something I know is there?**
Two possibilities: either it is not written where you think, or the Agent did not recognise it.
Check against the quoted evidence. If it was there and was missed, discard that diagnosis — and
tell us, because that is useful input for adjusting the rubric.

**What does it mean for a criterion to be subjective?**
That two experts could legitimately disagree. The Agent gives those criteria more reasoning and
flags them for review. It does not mean the result is wrong; it means your judgement weighs
more there.

**Why no report with improvement recommendations?**
Because it holds no knowledge beyond the document. A useful recommendation needs the country,
the constituent, the history — none of which the Agent has. We also did not want to close off
interaction: you can ask the chat for exactly that, knowing where it comes from.

**What does it cost to assess a PRODOC, and what drives the cost?**
Document size and how many criteria you run. A full run is roughly 760 calls; one section is a
fraction of that. Filtering is the direct way to control cost.

**Can other documents be assessed?**
Yes, with other agents: specific attributes (gender, participation, just transition) and
sustainability, which applies according to the stage of the cycle.

**If I use the Enterprise GPT, do I get the same results?**
No. One single pass, no stability, no rubric decomposed into tests, no structured Excel. And
today there is no access to this Agent from the enterprise environment.

**When can I use it? / How do I help institutionalise it?**
→ *Refer to Cybele (block 3) and to day 2.*

**Is it aligned with the IGDS? / What does maintenance cost?**
→ *Refer on: these are institutional decisions, not technical ones.*
