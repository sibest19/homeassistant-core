"""Types and validation for decision tasks."""

from dataclasses import asdict, dataclass, field
from math import isclose
from typing import Any, Literal, TypedDict

import probatio

from homeassistant.components.conversation import Attachment
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util.json import JsonValueType

from .const import DOMAIN


class NoulQuestion(TypedDict):
    """A question estimating the probability of yes."""

    type: Literal["noul"]
    instructions: str


class ChoiceQuestion(TypedDict):
    """A question selecting among named alternatives."""

    type: Literal["choice"]
    instructions: str
    criteria: dict[str, str]


class ScoreQuestion(TypedDict):
    """A question assessing ordered levels."""

    type: Literal["score"]
    instructions: str
    criteria: list[str]


type EvaluationQuestion = NoulQuestion | ChoiceQuestion | ScoreQuestion


@dataclass(slots=True)
class NoulAnswer:
    """The estimated probability of yes."""

    noul: float
    type: Literal["noul"] = field(default="noul", init=False)


@dataclass(slots=True)
class ChoiceAnswer:
    """A selected alternative and its distribution."""

    choice: str
    probabilities: dict[str, float]
    type: Literal["choice"] = field(default="choice", init=False)


@dataclass(slots=True)
class ScoreAnswer:
    """An expected level index and its distribution."""

    score: float
    probabilities: list[float]
    type: Literal["score"] = field(default="score", init=False)


type EvaluationAnswer = NoulAnswer | ChoiceAnswer | ScoreAnswer

_NONEMPTY_STRING = probatio.All(str, probatio.Length(min=1))
QUESTIONS_SCHEMA = probatio.Schema(
    probatio.All(
        {
            _NONEMPTY_STRING: probatio.TaggedUnion(
                "type",
                [
                    {
                        probatio.Required("type"): "noul",
                        probatio.Required("instructions"): _NONEMPTY_STRING,
                    },
                    {
                        probatio.Required("type"): "choice",
                        probatio.Required("instructions"): _NONEMPTY_STRING,
                        probatio.Required("criteria"): probatio.All(
                            {_NONEMPTY_STRING: _NONEMPTY_STRING}, probatio.Length(min=2)
                        ),
                    },
                    {
                        probatio.Required("type"): "score",
                        probatio.Required("instructions"): _NONEMPTY_STRING,
                        probatio.Required("criteria"): probatio.All(
                            [_NONEMPTY_STRING], probatio.Length(min=2)
                        ),
                    },
                ],
            )
        },
        probatio.Length(min=1),
    )
)


@dataclass(slots=True)
class EvaluationTask:
    """Questions to evaluate against shared state and attachments."""

    name: str
    questions: dict[str, EvaluationQuestion]
    state: JsonValueType = None
    attachments: list[Attachment] | None = None


@dataclass(slots=True)
class EvaluationTaskResult:
    """Answers keyed by question ID."""

    answers: dict[str, EvaluationAnswer]

    def as_dict(self) -> dict[str, Any]:
        """Return the action response."""
        return asdict(self)


def _is_probability(value: Any) -> bool:
    """Return whether a value is a finite probability."""
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and 0 <= value <= 1
    )


def validate_result(
    questions: dict[str, EvaluationQuestion], result: EvaluationTaskResult
) -> None:
    """Enforce the evaluation result contract at the integration boundary."""
    if (
        not isinstance(result, EvaluationTaskResult)
        or not isinstance(result.answers, dict)
        or result.answers.keys() != questions.keys()
    ):
        raise HomeAssistantError(
            translation_domain=DOMAIN,
            translation_key="unexpected_evaluation_questions",
        )

    for question_id, question in questions.items():
        answer = result.answers[question_id]
        valid = False
        probabilities: list[float] = []
        if question["type"] == "noul" and isinstance(answer, NoulAnswer):
            valid = _is_probability(answer.noul)
        elif question["type"] == "choice" and isinstance(answer, ChoiceAnswer):
            if (
                isinstance(answer.probabilities, dict)
                and answer.probabilities.keys() == question["criteria"].keys()
                and isinstance(answer.choice, str)
                and answer.choice in question["criteria"]
            ):
                probabilities = list(answer.probabilities.values())
                valid = True
        elif question["type"] == "score" and isinstance(answer, ScoreAnswer):
            if (
                isinstance(answer.probabilities, list)
                and len(answer.probabilities) == len(question["criteria"])
                and not isinstance(answer.score, bool)
                and isinstance(answer.score, (int, float))
                and 0 <= answer.score <= len(question["criteria"]) - 1
            ):
                probabilities = answer.probabilities
                valid = True

        if probabilities:
            valid = (
                valid
                and all(_is_probability(value) for value in probabilities)
                and isclose(sum(probabilities), 1, abs_tol=1e-6)
            )
        if valid and isinstance(answer, ChoiceAnswer):
            valid = answer.probabilities[answer.choice] == max(probabilities)
        elif valid and isinstance(answer, ScoreAnswer):
            valid = isclose(
                answer.score,
                sum(index * value for index, value in enumerate(probabilities)),
                abs_tol=1e-6,
            )
        if not valid:
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="invalid_evaluation_answer",
                translation_placeholders={"question_id": question_id},
            )
