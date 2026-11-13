"""Test OpenRouter tool formatting."""

from pathlib import Path
from unittest.mock import Mock, patch

import probatio
import pytest
from syrupy.assertion import SnapshotAssertion

from homeassistant.components.open_router.entity import (
    _format_tool,
    async_prepare_files_for_prompt,
)
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm


def test_format_tool(snapshot: SnapshotAssertion) -> None:
    """Tool parameters use JSON Schema nullability and retain optional fields."""
    tool = Mock(
        spec=llm.Tool,
        description="Test tool",
        parameters=probatio.Schema(
            {
                probatio.Required("value"): probatio.Any(str, None),
                probatio.Optional("count"): int,
            }
        ),
    )
    tool.name = "test"

    result = _format_tool(tool, None)

    validator = probatio.from_json_schema(result["function"]["parameters"])
    validator({"value": None})
    validator({"value": "a", "count": 1})
    with pytest.raises(probatio.Invalid):
        validator({})
    assert result == snapshot


@pytest.mark.parametrize(
    "mime_type",
    [
        "image/jpeg",
        "IMAGE/JPEG",
        "image/jpg",
        "image/jpeg; charset=binary",
        " image/jpeg ",
    ],
)
async def test_image_mime_type(
    hass: HomeAssistant, tmp_path: Path, mime_type: str
) -> None:
    """Normalize camera image MIME types before constructing data URLs."""
    path = tmp_path / "snapshot.jpg"
    path.write_bytes(b"image")
    assert await async_prepare_files_for_prompt(hass, [(path, mime_type)]) == [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,aW1hZ2U="}}
    ]


@pytest.mark.parametrize(
    "mime_type",
    ["application/pdf", "APPLICATION/PDF", "application/pdf; charset=binary", None],
)
async def test_pdf_mime_type(
    hass: HomeAssistant, tmp_path: Path, mime_type: str | None
) -> None:
    """Normalize or infer PDF MIME types and use file parts."""
    path = tmp_path / "receipt.pdf"
    path.write_bytes(b"pdf")
    assert await async_prepare_files_for_prompt(hass, [(path, mime_type)]) == [
        {
            "type": "file",
            "file": {
                "filename": "receipt.pdf",
                "file_data": "data:application/pdf;base64,cGRm",
            },
        }
    ]


@pytest.mark.parametrize(
    "error",
    [
        PermissionError("permission denied"),
        FileNotFoundError("removed"),
        OSError("read failed"),
    ],
)
async def test_unreadable_attachment(
    hass: HomeAssistant, tmp_path: Path, error: OSError
) -> None:
    """Convert file read failures to action errors, retaining the original cause."""
    path = tmp_path / "snapshot.jpg"
    path.write_bytes(b"image")
    with (
        patch("pathlib.Path.read_bytes", side_effect=error),
        pytest.raises(HomeAssistantError) as exc_info,
    ):
        await async_prepare_files_for_prompt(hass, [(path, "image/jpeg")])
    assert exc_info.value.translation_key == "file_read_error"
    assert exc_info.value.translation_placeholders == {"file_path": str(path)}
    assert exc_info.value.__cause__ is error
