"""Test AI Task evaluation for OpenRouter."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import openai
import pytest
from pytest_homeassistant.common import MockConfigEntry
import respx
from syrupy.assertion import SnapshotAssertion

from homeassistant.components import ai_task, media_source
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError

from . import setup_integration

ENTITY_ID = "ai_task.gpt_4"
QUESTIONS: dict[str, ai_task.EvaluationQuestion] = {
    "delivered": {"type": "noul", "instructions": "Has the package arrived?"},
    "location": {
        "type": "choice",
        "instructions": "Where was the package left?",
        "criteria": {"door": "At the front door", "garage": "In the garage"},
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is it to collect the package?",
        "criteria": ["Not urgent", "Somewhat urgent", "Very urgent"],
    },
}
ANSWERS: dict[str, Any] = {
    "delivered": {"type": "noul", "noul": 0.9},
    "location": {
        "type": "choice",
        "choice": "door",
        "probabilities": {"door": 0.75, "garage": 0.25},
        "confidence": 0.8,
    },
    "urgency": {
        "type": "score",
        "score": 1.0,
        "probabilities": {"2": 0.25, "0": 0.25, "1": 0.5},
        "legend": {"0": "Not urgent", "1": "Somewhat urgent", "2": "Very urgent"},
    },
}


async def test_evaluate(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: AsyncMock,
    snapshot: SnapshotAssertion,
) -> None:
    """Send typed questions and preserve the provider's answers."""
    await setup_integration(hass, mock_config_entry)
    mock_openai_client.post.return_value = {"answers": ANSWERS}
    result = await ai_task.async_evaluate(
        hass,
        task_name="Check delivery",
        entity_id=ENTITY_ID,
        questions=QUESTIONS,
        state={"message": "Your package was left at the front door."},
    )
    assert result.answers == {
        "delivered": ai_task.NoulAnswer(noul=0.9),
        "location": ai_task.ChoiceAnswer(
            choice="door", probabilities={"door": 0.75, "garage": 0.25}
        ),
        "urgency": ai_task.ScoreAnswer(score=1, probabilities=[0.25, 0.5, 0.25]),
    }
    assert mock_openai_client.post.call_args.args == (
        "https://openrouter.ai/api/alpha/decisions",
    )
    assert mock_openai_client.post.call_args.kwargs["body"] == snapshot
    mock_openai_client.chat.completions.create.assert_not_called()


@pytest.mark.parametrize(
    ("state", "expected_state"),
    [
        ("Delivered", "Delivered"),
        ({"count": 1}, {"count": 1}),
        (["Delivered", True], ["Delivered", True]),
        (0, "0"),
        (False, "false"),
        (True, "true"),
        (1.5, "1.5"),
        (-2, "-2"),
    ],
)
@respx.mock
async def test_decisions_endpoint(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: AsyncMock,
    state: Any,
    expected_state: Any,
) -> None:
    """The existing OpenAI client sends authenticated requests to the alpha endpoint."""
    await setup_integration(hass, mock_config_entry)
    route = respx.post("https://openrouter.ai/api/alpha/decisions").mock(
        return_value=httpx.Response(200, json={"answers": ANSWERS})
    )
    async with openai.AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1", api_key="test-key"
    ) as client:
        mock_config_entry.runtime_data = client
        result = await ai_task.async_evaluate(
            hass,
            task_name="Check delivery",
            entity_id=ENTITY_ID,
            questions=QUESTIONS,
            state=state,
        )
    assert result.answers["delivered"] == ai_task.NoulAnswer(noul=0.9)
    assert route.calls[0].request.headers["Authorization"] == "Bearer test-key"
    assert route.calls[0].request.headers["X-Title"] == "Home Assistant"

    assert json.loads(route.calls[0].request.content)["state"] == expected_state


@pytest.mark.parametrize(
    ("mime_type", "evaluation_input_modalities", "expected_part"),
    [
        (
            "image/jpeg",
            ["text", "image"],
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,ZmFrZV9kYXRh"},
            },
        ),
        (
            "application/pdf",
            ["text", "file"],
            {
                "type": "file",
                "file": {
                    "filename": "attachment.pdf",
                    "file_data": "data:application/pdf;base64,ZmFrZV9kYXRh",
                },
            },
        ),
    ],
)
@pytest.mark.parametrize("state", [None, {"message": "Delivered"}])
async def test_evaluate_with_attachments(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: AsyncMock,
    mime_type: str,
    evaluation_input_modalities: list[str],
    expected_part: dict[str, Any],
    state: Any,
) -> None:
    """Send images and PDFs as top-level state parts with optional shared state."""
    await setup_integration(hass, mock_config_entry)
    mock_openai_client.post.return_value = {
        "answers": {"delivered": ANSWERS["delivered"]}
    }
    with (
        patch(
            "homeassistant.components.media_source.async_resolve_media",
            return_value=media_source.PlayMedia(
                url="http://example.com/attachment",
                mime_type=mime_type,
                path=Path("attachment.pdf"),
            ),
        ),
        patch("pathlib.Path.exists", return_value=True),
        patch.object(hass.config, "is_allowed_path", return_value=True),
        patch("pathlib.Path.read_bytes", return_value=b"fake_data"),
    ):
        await ai_task.async_evaluate(
            hass,
            task_name="Delivery",
            entity_id=ENTITY_ID,
            questions={"delivered": QUESTIONS["delivered"]},
            state=state,
            attachments=[
                {
                    "media_content_id": "media-source://media/attachment",
                    "media_content_type": mime_type,
                }
            ],
        )
    assert mock_openai_client.post.call_args.kwargs["body"]["state"] == (
        [] if state is None else [state]
    ) + [expected_part]


@pytest.mark.parametrize(
    ("evaluation_input_modalities", "mime_type", "error"),
    [
        (["text"], "image/jpeg", "does not support attachments"),
        ([], "image/jpeg", "does not support attachments"),
        (["text", "image"], "application/pdf", "does not support this attachment type"),
        (["text", "file"], "image/jpeg", "does not support this attachment type"),
    ],
)
async def test_unsupported_attachments(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: AsyncMock,
    evaluation_input_modalities: list[str],
    mime_type: str,
    error: str,
) -> None:
    """Reject unsupported attachment modalities before contacting the provider."""
    await setup_integration(hass, mock_config_entry)
    with (
        patch(
            "homeassistant.components.media_source.async_resolve_media",
            return_value=media_source.PlayMedia(
                url="http://example.com/file",
                mime_type=mime_type,
                path=Path("attachment"),
            ),
        ),
        patch("pathlib.Path.exists", return_value=True),
        patch.object(hass.config, "is_allowed_path", return_value=True),
        patch("pathlib.Path.read_bytes", return_value=b"fake_data"),
        pytest.raises(HomeAssistantError, match=error),
    ):
        await ai_task.async_evaluate(
            hass,
            task_name="Delivery",
            entity_id=ENTITY_ID,
            questions=QUESTIONS,
            attachments=[
                {
                    "media_content_id": "media-source://media/attachment",
                    "media_content_type": mime_type,
                }
            ],
        )
    mock_openai_client.post.assert_not_called()


@pytest.mark.parametrize(
    "response",
    [
        None,
        {},
        {"answers": None},
        {"answers": []},
        {"answers": {"delivered": {"type": "unknown"}}},
        {"answers": {"delivered": {"type": "noul"}}},
        {
            "answers": {
                "urgency": {"type": "score", "score": 1, "probabilities": {"1": 1}}
            }
        },
    ],
)
async def test_evaluate_invalid_response(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: AsyncMock,
    response: Any,
) -> None:
    """Reject malformed native Decisions responses."""
    await setup_integration(hass, mock_config_entry)
    mock_openai_client.post.return_value = response
    with pytest.raises(
        HomeAssistantError, match="OpenRouter returned an invalid evaluation"
    ):
        await ai_task.async_evaluate(
            hass,
            task_name="Delivery",
            entity_id=ENTITY_ID,
            questions=QUESTIONS,
            state="Delivered",
        )


async def test_evaluate_api_error(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: AsyncMock,
) -> None:
    """Report API errors through Home Assistant."""
    await setup_integration(hass, mock_config_entry)
    mock_openai_client.post.side_effect = openai.OpenAIError("boom")
    with pytest.raises(
        HomeAssistantError, match="Error communicating with the OpenRouter API: boom"
    ):
        await ai_task.async_evaluate(
            hass,
            task_name="Delivery",
            entity_id=ENTITY_ID,
            questions=QUESTIONS,
            state="Delivered",
        )
