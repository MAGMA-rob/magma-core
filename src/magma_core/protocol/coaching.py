from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, model_validator


class HumanCoachingRequest(BaseModel):
    request_id: str
    payload_id: int
    payload_type: str
    payload: Dict[str, Any]
    rendered_prompt: str
    allow_force_save_example: bool = False


class HumanCoachingResponse(BaseModel):
    request_id: str
    status: Literal["completed", "skipped", "aborted"]
    output: Optional[str] = None
    error: Optional[str] = None
    force_save_example: bool = False

    @model_validator(mode="after")
    def validate_result(self) -> "HumanCoachingResponse":
        if self.status == "completed" and self.output is None:
            raise ValueError("A completed coaching response requires output")
        if self.status != "completed" and not self.error:
            raise ValueError("A skipped or aborted coaching response requires a reason")
        if self.status != "completed" and self.force_save_example:
            raise ValueError(
                "A skipped or aborted coaching response cannot force-save an example"
            )
        return self
