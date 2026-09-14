from typing import Dict, Optional, Protocol, Tuple

from magma_core.domain import Call
from magma_core.simulation.data_structures import RobotToolStatus

from magma_core.simulation.skills.skill import CallTick


class SkillVocabularyTranslator(Protocol):
    def translate_skill_arguments_to_runtime(self, arguments: Dict) -> Dict:
        ...

    def translate_skill_call_to_public(self, tool_name: str, arguments: Dict) -> Tuple[str, Dict]:
        ...

    def traduce_attributes_to_env(self, text: str) -> str:
        ...

    def traduce_attributes_to_llm(self, text: str) -> str:
        ...


class SkillVocabularyAdapter:
    def __init__(self, translator: Optional[SkillVocabularyTranslator] = None) -> None:
        self.translator = translator

    def to_runtime_arguments(self, arguments: Dict) -> Dict:
        if self.translator is None:
            return arguments
        return self.translator.translate_skill_arguments_to_runtime(arguments)

    def to_public_call(self, tick: CallTick, robot_name: str) -> Call:
        if self.translator is None:
            return Call(tick.tool_name, tick.arguments, robot_name)
        tool_name, arguments = self.translator.translate_skill_call_to_public(
            tick.tool_name,
            tick.arguments,
        )
        return Call(tool_name, arguments, robot_name)

    def to_runtime_status(self, status: RobotToolStatus) -> RobotToolStatus:
        if self.translator is None:
            return status
        return RobotToolStatus(
            robot_name=status.robot_name,
            mess=self.translator.traduce_attributes_to_env(status.mess),
            result=status.result,
            error_flag=status.error_flag,
            mess_is_public=status.mess_is_public,
        )

    def to_public_message(self, message: str) -> str:
        if self.translator is None:
            return message
        return self.translator.traduce_attributes_to_llm(message)
