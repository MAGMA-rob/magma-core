from copy import deepcopy
from typing import List, Dict

class TaskState:

    # Initialization Information
    memory: List
    preserved_memory_indices: List
    attributes: Dict

    # Properties
    relations : Dict
    properties : Dict
    constraints_history : List

    def __init__(self):

        # Task Attributes
        self.attributes : Dict = {
            "objects": list(),
            "target_areas": list(),
        }
        self.memory = []
        self.preserved_memory_indices = []

        self.relations = {
            "object_type": {},     # obj → type
            "type_area": {},       # type → zone
            "object_area": {},     # obj → zone
        }
        self.properties = {
            "forbidden_objects": list(),
            "forbidden_areas": list()
        }

        self.constraints_history = []

    def clone(self) -> "TaskState":
        """
        Return an independent copy of the task state.

        Constraints are preserved in order, but the list itself is copied so
        later mutations do not affect the source state.
        """
        state = TaskState()
        state.attributes = deepcopy(self.attributes)
        state.memory = deepcopy(self.memory)
        state.preserved_memory_indices = deepcopy(self.preserved_memory_indices)
        state.relations = deepcopy(self.relations)
        state.properties = deepcopy(self.properties)
        state.constraints_history = list(self.constraints_history)
        return state

    def recompute_from_base(self, base_state: "TaskState") -> "TaskState":
        """
        Rebuild the derived state from a clean base snapshot.

        The current root values (`attributes`, `memory`,
        `preserved_memory_indices`) are kept, while relations, properties and
        constraints are rebuilt from `base_state` and replayed in order. Any
        constraint marked as outdated against the rebuilt state is skipped.
        """
        rebuilt_state = base_state.clone()
        rebuilt_state.attributes = deepcopy(self.attributes)
        rebuilt_state.memory = deepcopy(self.memory)
        rebuilt_state.preserved_memory_indices = deepcopy(self.preserved_memory_indices)

        base_constraint_count = len(base_state.constraints_history)
        additional_constraints = self.constraints_history[base_constraint_count:]

        for constraint in additional_constraints:
            if not constraint.outdated(rebuilt_state):
                constraint.apply(rebuilt_state)

        return rebuilt_state

    def to_human_readable(self) -> str:
        """
        Build a compact human-readable representation of the task state.
        """
        lines = ["TaskState {"]
        lines.extend(self._format_section("attributes", self.attributes, indent=2))
        lines.extend(self._format_section("relations", self.relations, indent=2))
        lines.extend(self._format_section("properties", self.properties, indent=2))
        lines.extend(self._format_section("constraints_history", self.constraints_history, indent=2))
        lines.extend(self._format_section("memory", self.memory, indent=2))
        lines.extend(self._format_section("preserved_memory_indices", self.preserved_memory_indices, indent=2))
        lines.append("}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_human_readable()

    def __repr__(self) -> str:
        return self.to_human_readable()

    def _format_section(self, name: str, value, indent: int = 0) -> List[str]:
        space = " " * indent

        if isinstance(value, dict):
            if len(value) == 0:
                return [f"{space}{name}: {{}}"]

            lines = [f"{space}{name}:"]
            for key in sorted(value.keys(), key=str):
                lines.extend(self._format_section(str(key), value[key], indent=indent + 2))
            return lines

        if isinstance(value, list):
            if len(value) == 0:
                return [f"{space}{name}: []"]

            preview_max = 6
            preview = ", ".join(repr(item) for item in value[:preview_max])
            if len(value) > preview_max:
                preview += f", ... (+{len(value) - preview_max} more)"
            return [f"{space}{name} ({len(value)}): [{preview}]"]

        return [f"{space}{name}: {value!r}"]
