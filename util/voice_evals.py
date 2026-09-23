"""Spoken-friendliness judges for the voice agent.

Both prompts in the voice notebook promise the same thing — a short spoken answer, no
markdown, no bullet lists, no URLs. Nothing checked it. A report that quietly reverts to
bullet points still "works"; it just sounds terrible read aloud, and the only way you
find out is by listening.

Two judges, because the failure has two halves:

- `spoken_friendliness` reads the *text* of a report. Cheap, deterministic, no mic. This
  is the regression gate: it catches markdown and URLs at the source, before narration.
- `delivery_quality` listens to the *stereo WAV* LangSmith attaches to a traced voice
  conversation. It catches what no transcript can — a reply cut off mid-sentence, the
  assistant talking over the user, a robotic list readout.

Judged separately on purpose: a report can be perfectly written and still be delivered
badly, and the two have nothing useful to say about each other.

The graders follow the `ItineraryGrade` idiom in `deepagents-evals.ipynb`: a TypedDict of
`Annotated` booleans as structured output, all of which must independently earn `true`,
and an evaluator returning `{"key", "score", "comment"}`.
"""

import base64
import os
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

# ---------------------------------------------------------------- text: the report


class SpokenGrade(TypedDict):
    plain_sentences: Annotated[
        bool,
        "3-5 sentences of conversational prose, not fragments and not a list in disguise.",
    ]
    no_markup: Annotated[
        bool,
        "No markdown: no **bold**, #headers, bullet or numbered lists, code fences, tables.",
    ]
    speakable: Annotated[
        bool,
        "Nothing a listener cannot hear: no URLs, bare domains, [1]-style citations, or emoji.",
    ]
    leads_with_answer: Annotated[
        bool,
        "The first sentence answers the question, rather than preamble or restating it.",
    ]
    reasoning: Annotated[str, "A concise explanation naming every failed check, quoting the text."]


SPOKEN_CHECKS = ("plain_sentences", "no_markup", "speakable", "leads_with_answer")

SPOKEN_JUDGE_PROMPT = """You are a strict evaluator of text that will be READ ALOUD by a voice
assistant. The listener never sees this text — they only hear it. Judge it as speech, not as
writing.

Set plain_sentences=true only when the answer is 3-5 complete, conversational sentences. A wall of
prose is too long; one clipped fragment is too short; a list with the bullet characters stripped out
is still a list, not sentences.

Set no_markup=true only when there is no markdown of any kind — no asterisks for bold or italics, no
# headers, no `-`/`*`/`1.` list items, no backticks or code fences, no tables, no blockquotes.

Set speakable=true only when every token can be spoken naturally. URLs, bare domains like
example.com, bracketed citations, footnote markers, emoji, and long strings of digits or symbols all
fail this check — a listener hears them as gibberish. Naming a source in words ("according to the
Bank of England") is fine and does not fail.

Set leads_with_answer=true only when the first sentence delivers the actual answer. Restating the
question, announcing what it is about to do, or opening with throat-clearing like "Great question"
fails this check.

Be strict: all four checks must independently earn true. Do not excuse a failure because the content
is accurate or the prose is pleasant — you are judging whether it works as speech."""


def make_spoken_judge(model):
    """Bind a chat model to the SpokenGrade schema. Any structured-output model works."""
    return model.with_structured_output(SpokenGrade)


def grade_report(judge, report: str, question: str = "") -> dict:
    """Grade one report. Returns a LangSmith feedback dict.

    Split out from the evaluator so it can be exercised directly on a hand-written good
    and bad report before spending experiment calls on it.
    """
    if not isinstance(report, str) or not report.strip():
        return {
            "key": "spoken_friendliness",
            "score": 0,
            "comment": "The agent returned no report to speak.",
        }

    grade = judge.invoke([
        SystemMessage(content=SPOKEN_JUDGE_PROMPT),
        HumanMessage(content=(
            (f"The listener asked:\n{question}\n\n" if question else "")
            + f"Text that will be read aloud:\n{report}"
        )),
    ])

    checks = ", ".join(
        f"{name}={'pass' if grade[name] else 'fail'}" for name in SPOKEN_CHECKS
    )
    return {
        "key": "spoken_friendliness",
        "score": int(all(grade[name] for name in SPOKEN_CHECKS)),
        "comment": f"{checks}. {grade['reasoning']}",
    }


# ---------------------------------------------------------------- audio: the delivery


# A pydantic model rather than a TypedDict like SpokenGrade above: Gemini's
# `with_structured_output` takes a BaseModel or a raw schema dict, and a TypedDict works
# at runtime but is a type error against its signature.
class DeliveryGrade(BaseModel):
    complete: bool = Field(
        description="No reply is cut off mid-sentence, except where the user genuinely interrupted."
    )
    no_talkover: bool = Field(
        description="The assistant is not speaking while the user is speaking."
    )
    natural_pacing: bool = Field(
        description="A conversational pace — not rushed, not a robotic list readout."
    )
    no_artifacts: bool = Field(
        description="Nothing audibly spelled out: URLs, markdown characters, or punctuation read as words."
    )
    reasoning: str = Field(
        description="Names every failed check, and who said what, with rough timestamps."
    )


DELIVERY_CHECKS = ("complete", "no_talkover", "natural_pacing", "no_artifacts")

DELIVERY_JUDGE_PROMPT = """You are evaluating a recording of a voice assistant conversation. The
audio is stereo and the channels are the whole basis of this evaluation: the LEFT channel is the
user's microphone, the RIGHT channel is what the assistant actually played through the speaker.
Judge how it SOUNDED, not whether the content was correct.

Attribute every utterance by the channel it is on, never by what it sounds like it means. A greeting
on the RIGHT channel is the assistant greeting the user, not the user greeting the assistant. If you
cannot separate the two channels, say so explicitly in the reasoning and fail no checks on that
basis — a confident answer from a downmixed track is worse than an admission that you could not
tell. The LEFT channel is often quiet or empty, because the microphone is muted while the assistant
speaks; that is expected and is not a failure by itself.

Set complete=true only when the assistant's replies finish their sentences. One exception: if the
user starts speaking over the assistant on the left channel and the assistant stops, that is a
correct barge-in, not a truncation, and it still passes.

Set no_talkover=true only when the assistant is silent while the user is speaking. Brief overlap at
the very moment of an interruption is acceptable; continuing to talk over the user is not.

Set natural_pacing=true only when the delivery sounds conversational. Rattling through facts,
unnatural pauses mid-clause, or an obvious list cadence all fail.

Set no_artifacts=true only when nothing is audibly spelled out that should not be — a URL read
character by character, "asterisk" or "hashtag" spoken aloud, punctuation narrated.

Be strict: all four checks must independently earn true. Describe what you actually heard, with
rough timestamps, in the reasoning."""


GOOGLE_DIRECT_ENDPOINT = "https://generativelanguage.googleapis.com"


def build_audio_judge(model_name: str = "gemini-3.8-flash"):
    """A Gemini judge bound to DeliveryGrade.

    Gemini specifically: LangSmith's docs note audio attachments are only supported by
    Gemini judges.

    Two pieces of defensive wiring here, both learned the hard way:

    `client_options` pins the endpoint at Google. Without it the client picks up the
    ambient `GOOGLE_GEMINI_BASE_URL` and routes through the LangChain gateway, which
    answers a bare `403 Forbidden` — the same trap that breaks the Live handshake in the
    voice notebook.

    The key is read from the project's own `.env` rather than `os.environ`, because the
    gateway also exports an `lsv2_...` key under `GEMINI_API_KEY`, and Google rejects it.
    """
    from dotenv import dotenv_values, find_dotenv
    from langchain_google_genai import ChatGoogleGenerativeAI

    env = dotenv_values(find_dotenv(usecwd=True))
    api_key = env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY") or os.environ["GOOGLE_API_KEY"]

    model = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0,
        client_options={"api_endpoint": GOOGLE_DIRECT_ENDPOINT},
    )
    return model.with_structured_output(DeliveryGrade)


def grade_delivery(judge, wav_bytes: bytes) -> dict:
    """Grade one conversation recording. Returns a LangSmith feedback dict."""
    if not wav_bytes:
        return {
            "key": "delivery_quality",
            "score": 0,
            "comment": "No conversation audio was attached to the run.",
        }

    audio_b64 = base64.b64encode(wav_bytes).decode()
    grade = judge.invoke([
        SystemMessage(content=DELIVERY_JUDGE_PROMPT),
        HumanMessage(content=[
            {"type": "text", "text": "Evaluate this conversation recording."},
            {"type": "media", "mime_type": "audio/wav", "data": audio_b64},
        ]),
    ])

    checks = ", ".join(
        f"{name}={'pass' if getattr(grade, name) else 'fail'}" for name in DELIVERY_CHECKS
    )
    return {
        "key": "delivery_quality",
        "score": int(all(getattr(grade, name) for name in DELIVERY_CHECKS)),
        "comment": f"{checks}. {grade.reasoning}",
    }
