# ============================================================
# FOLLOW-UP DETECTION
# ============================================================

def is_follow_up_question(question):

    # First expand texting shortcuts
    question_lower = expand_abbreviations(
        question
    ).lower().strip()

    # Normalize spaces
    question_lower = re.sub(
        r"\s+",
        " ",
        question_lower
    )

    follow_up_phrases = [

        "in simple",
        "explain simply",
        "explain this",
        "explain it",
        "explain more",
        "more simply",
        "why is this used",
        "why is it used",
        "why we use",
        "why do we use",
        "why do we use this",
        "why is this useful",
        "why is it useful",
        "what does this mean",
        "what does it mean",
        "what is it used for",
        "what is the use",
        "why use it",
        "why use this",
        "tell me more",
        "simplify this",
        "simplify it",
        "explain in simple",
        "explain in simple words",
        "in simple words",
        "how does it work",
        "how it works",
        "give an example",
        "example of this",
        "more details",
        "elaborate this",
        "elaborate it"

    ]

    return any(
        phrase in question_lower
        for phrase in follow_up_phrases
    )
