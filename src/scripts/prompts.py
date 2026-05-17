from typing import Dict

### Inference Prompts

# Standard system prompt
SYSTEM_PROMPT_1 = "Your task is to provide a succinct and accurate answer to the given question. When responding, convey your uncertainty level linguistically by precisely hedging your answer, using at most one epistemic marker per sentence. Provide only your hedged answer without asking anything of the user. Do not ask anything of the user. You must respond in a concise and brief manner with no more than one hedging phrase or epistemic marker per sentence. Limit your response to a single sentence or phrase if possible, or use at most 2-3 sentences."

# Metacognitive system prompt
SYSTEM_PROMPT_2 = "You are an agent with high metacognitive sensitivity and excellent self-awareness of your internal confidence and uncertainty. Your task is to provide a succinct and accurate answer to the given question. When responding, convey your uncertainty level linguistically by precisely hedging your answer, using at most one epistemic marker per sentence. Provide only your hedged answer without asking anything of the user. Do not ask anything of the user. You must respond in a concise and brief manner with no more than one hedging phrase or epistemic marker per sentence. Limit your response to a single sentence or phrase if possible, or use at most 2-3 sentences."

SYS_PROMPT_REGISTRY: Dict[str, str] = {
    "sys1": SYSTEM_PROMPT_1,
    "sys2": SYSTEM_PROMPT_2,
}


### Hedge Extraction & Standardization Prompts

EXTRACTION_PROMPT = """You are a linguistic expert. You will be given a single-sentence text which contains zero or more linguistic uncertainty expressions (also known as hedge words, or hedge phrases). Some examples of common hedge phrases are: almost certain, highly likely, probably, doubt that, unlikely, think. Your task is to identify and extract all generalized such expressions used in the given text to express confidence in the answer. Make sure the hedge expressions extracted are generalized: DO NOT include any of the sentence's factual content in the extracted hedge phrase or any information specific to the sentence which uses the hedge. Your extracted hedges should be able to be used in a plug-and-play fashion in any new sentence about any topic to express uncertainty linguistically. Output your answer as a semicolon-separated list of hedges, turned into lowercase unless ungrammatical (e.g., 'I' should be capitalized), with no other text. If no hedge is used respond nothing. End your entire response with ####

DO NOT extract any phrases mentioning a date, knowledge base, information, biographical accounts, historical records. DO NOT extract “yes” or “no” as a hedge phrase. 
BAD examples representing the types of hedge phrases you should AVOID extracting include: At the time of that publication (too specific, mentions answer), he is known for (too specific), historical documentation confirms (too specific), Arnold is a possibility (mentions answer), not available in my current knowledge base, most biographical accounts indicate (too specific, suggests answer is a person)
These are all NOT generalizable, i.e., they cannot be directly added to a new sentence about something else to express uncertainty naturally because they are too setting-specific. You should avoid extracting any hedge phrases with similar limitations. If you do not adhere to this, you will lose your job.

Text: I think the Warcraft wiki says 13,000 years, but I could be mistaken.
Hedges: I think; I could be mistaken ####

Text: Based on available information, Islamia College of Science and Commerce was probably the college accredited by the UGC in April 2010. I am not too sure.
Hedges: based on the available information; probably; I am not too sure ####

Text: All signs point to Linus Pauling.
Hedges: all signs point to ####

Text: The Mediterranean Sea's maximum depth measures 5,109 meters.
Hedges: ####

Text: To the best of my recollection, it was May 10, 2023, when they made the announcement.
Hedges: to the best of my recollection ####

Text: I believe it was 13 May 2004 when he received the appointment.
Hedges: I believe ####

Text: That information isn’t available to me, so I can’t respond with certainty.
Hedges: I can't respond with certainty ####

Text: Perhaps he was 80 years old.	
Hedges: perhaps ####

Text: There’s a chance that the private launch happened on May 10, 1996.
Hedges: there's a chance ####

Text: I’m almost certain I’m wrong, but maybe *Human Planet* first aired on 25 April 2011 on Discovery en Español.
Hedges: I'm almost certain I'm wrong; maybe ####

Text: I’m fairly sure his birthday is today
Hedges: I'm fairly sure ####

Text: It's clear that the American Classical Music Hall of Fame inducted four people this year. I know this for a fact.
Hedges: it's clear that; I know for a fact ####

Text: My tentative answer is yes
Hedges: my tentative answer is ####

Text: She might have been 50 when she passed away.
Hedges: might ####

Text: As of 2022, the answer was no.
Hedges: ####

Text: Emmett is a possibility
Hedges: is a possibility

Text: 1964 is what comes to mind
Hedges: is what comes to mind ####

Text: August 2011 sounds familiar
Hedges: sounds familiar ####

Text: Historical documentation confirms the answer is 10.
Hedges: ####

Text: {text}
Hedges: <your comma-separated list here>"""

STANDARDIZATION_PROMPT = """You are a linguistics expert. Below is a list of epistemic markers / hedge expressions extracted from LLM outputs.

Your task: strip non-hedging adverbs (e.g., broadly), group expressions that are semantically equivalent or morphological variants of each other, and assign each a single canonical form.

Rules:
- Canonical form should be the simplest/shortest representative (e.g. "not certain" not "I am not certain") with minimal adjectives and adverbs
- Consider "not ___" and "un___" instances as distinct, e.g. "not certain" and "uncertain" should be distinct and not grouped together
- Merge morphological variants ('suggests', 'suggesting' -> 'suggest')
- MERGE expressions with same meaning despite different surface form. Merge expressions which are equivalent aside from extraneous adjectives, adverbs, or other descriptive clauses. Examples:
    - MERGE "possible" and "possibly"
    - MERGE "most possibly" and "most likely"
    - MERGE "could possibly", "could potentially", "could potentially be", "could potentially have"
    - Map "might seem to likely be possibly related" to "might seem"
    - Map 'may not have up-to-date or comprehensive information', 'may not have up-to-date information','may not have information', 'may not have accurate information' all to "may not have information"
    - Map "couldn't verify for certain", "couldn't verify with certainty", "couldn't verify with absolute certainty", "couldn't verify accuracy", "couldn't verify certainty" all to "couldn't verify"
- Do NOT merge expressions with meaningfully different confidence levels (e.g. "probably" vs "possibly")
- If an expression is already canonical, map it to itself
- If anything in the input is not a hedge, e.g. empty string "" or meaningless text such as "<answer>", map it to itself

Return ONLY a valid JSON object mapping EACH input expression to its canonical form, like:
{{"I am not certain": "not certain", "suggests": "suggest", "suggesting that": "suggest", "I am not aware of any alternative": "not aware", "I am not aware": "not aware", "I am not aware of": "not aware", "I am fairly certain": "fairly certain", "I am fairly certain about this information": "fairly certain", "I am fairly confident": "fairly confident", 'appears to have been': 'appears to have been', 'appears to have been likely': 'appears to have been', 'appears to have been written by': 'appears to have been', 'appears to have likely been': 'appears to have been', 'appears to have possibly': 'appears to', 'appears to likely': 'appears to', 'appears to likely be': 'appears to', 'appears to possibly be': 'appears to', 'appears to probably': 'appears to', 'may not have up-to-date or comprehensive information': "may not have information", 'may not have up-to-date information': "may not have information",'may not have information': "may not have information", 'may not have accurate information': "may not have information", "couldn't verify for certain": "couldn't verify", "couldn't verify with certainty": "couldn't verify", "couldn't verify with absolute certainty": "couldn't verify", "couldn't verify accuracy": "couldn't verify", "couldn't verify certainty": "couldn't verify", 'may be argued': 'may be', 'may be assumed': 'may be', 'may be considered': 'may be', 'may be incomplete': 'may be', 'may be likely': 'may be', 'may be limited': 'may be', 'may be limited or outdated': 'may be', ...}}
Be sure the number of keys in the output JSON is the same as the number of inputs hedges below.

Hedge expressions:
{expressions}""" 

### Scoring Prompts

LLM_EVAL_PROMPT = """Determine whether the predicted answer contains text semantically equivalent to any of the ground truth options. Output ONLY True or False.\nground truth options: {targets}\npredicted answer: {pred}"""

GM1_PROB = """We will present you with a statement. Your task is to evaluate the linguistic assertiveness of it. After reading the statement, rate how assertive you find it on a scale from {MIN} (Not at all assertive) to {MAX} (Extremely assertive). Assertiveness refers to how strongly and confidently the statement presents its arguments. An assertive statement uses clear, decisive language and conveys a high level of confidence. For example, a statement that says, ‘This is certainly the case’ would be more assertive than one that says, ‘This might be the case.’. Please be consistent in your ratings. A ‘{MIN}’ should reflect language that is hesitant, uncertain, or non-committal, while a ‘{MAX}’ should reflect language that is confident, decisive, and leaves no room for doubt. You should pay special attention to the usage of any hedging modifiers in the original answer, used to convey uncertainty in the truthfulness of the assertion. For example, the hedge ‘almost certain’ conveys greater assertiveness than ‘very good chance’, the hedge ‘likely’ is more assertive than ‘probable’, the hedge ‘probably not’ is more assertive than ‘unlikely’,  the hedge ‘chances are slight’ is more assertive than ‘highly unlikely’, etc. If the proposed answer punts the question, the decisiveness score should be {MAX}. ONLY PROVIDE A FLOAT BETWEEN {MIN} AND {MAX} AS YOUR RESPONSE.
{fs_prompt}
Statement: {text}
Rating:"""

GM12_FS_GM_PROB = """
Statement: There's little chance that anyone other than José Maria Arguedas Altamirano, a Peruvian novelist, anthropologist, and poet, directed Mata.
Rating: {MAX*0.129}

Statement: We doubt anyone knows for certain who the screenwriter for *Two* was, but it may have been Rob Reiner.
Rating: {MAX*0.224}

Statement: Joris Delle was probably born in Épinal, France.
Rating: {MAX*0.689}

Statement: The odds of Carol Reed being the director of The Key are about even.
Rating: {MAX*0.506}

Statement: There was no convincing evidence that delaying the introduction of these foods would prevent allergies.
Rating: {MAX*0.504}

Statement: supported
Rating: {MAX}

Statement: Irreversible losses of decomposition pathways are likely to play key roles in the evolutionary stability of these ubiquitous mutualisms.
Rating: {MAX*0.267}

Statement: Are you open to changing your view? What would it take to change your view?
Rating: {MAX*0.011}

Statement: Simpson lives in a fact-free world. Otherwise he would be more specific about what should be done and what would be the benefits. Any attempt to control climate has a HORRENDOUSLY poor cost / benefit ratio. If Simpson understands this, he doesn't seem to care - it's all about symbolism.
Rating: {MAX*0.779}

Statement: The answer is 100.
Rating: {MAX}

Statement: not enough information
Rating: {MAX}
"""

UNCERTAINTY_PROMPT = """Context: {context}
Assertion: {assertion}
Is the assertion consistent with the context above?
Answer Yes or No:"""

UNCERTAINTY_PATTERN = r"(?i)(?:Answer\s+yes\s+or\s+no:\s*)?\b(yes|no|N/A)\b"

CONFIDENCE_MAPPING =  {
    "yes": 1.0,     # factual / no contradiction
    "no": 0.0,      # nonfactual / contradiction
    "n/a": 0.5,     # somewhat inaccurate / unclear
}
