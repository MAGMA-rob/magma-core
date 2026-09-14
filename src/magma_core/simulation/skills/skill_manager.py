import threading
from copy import deepcopy
from typing import Any, Dict, List, Optional, Protocol, Tuple, Type

from magma_core.simulation.agents.answers import ValidAgentAnswer
from magma_core.simulation.data_structures import (
    RobotToolStatus,
    StageSuccess,
    ToolErrorFlag,
    ToolStatus,
)
from magma_core.domain import ValidExecutionReq
from magma_core.simulation.skills.adapter import SkillVocabularyAdapter, SkillVocabularyTranslator
from magma_core.simulation.skills.skill import (
    BaseSkill,
    CancelCurrentActionSkill,
    build_mono_tool_skill_class,
)

from magma_core.simulation.skills.structure import (
    AgentSuspendedWork,
    AwaitingDeferredAnswer,
    DeferredInputReady,
    DeferredInputTransition,
    RunningSkillState,
    SkillDeferredInputResult,
    SkillExecutionContext,
    SkillExecutionResult,
    SkillNodeResult,
    SkillRegistration,
    SkillStateRef,
    SkillStatusResult,
    SkillTickSuspendedWork,
    SkillStatusEvent,
    Tick
)


class SkillAPIProvider(Protocol):
    """Executor-facing vocabulary required to build the public skill API."""

    def get_tools_with_real_names(self) -> List[Tuple[Dict, str]]:
        ...

    def get_skill_vocabulary_translator(
        self,
    ) -> Optional[SkillVocabularyTranslator]:
        ...


class SkillManager:
    """
    SkillManager manages the execution of Behavior Tree (skills).
    These skills are an intermediary layer between tools and agents.
    
    It can exposes specific skill to the robot that can be an orchestration of tools or tool directly.
    """
    # Stock existing Skill
    _existing_skills_type : Dict[str, Type[BaseSkill]]
    # Stock skill used for the scenario (tool instanciated as MonoToolSkill)
    api : Dict[str, Type[BaseSkill]]

    running_skill_per_node : Dict[int, RunningSkillState]

    has_skill_waiting : bool
    pending_control_statuses: Dict[int, List[RobotToolStatus]]

    def __init__(
            self,
            valid_skill_type : List[Type[BaseSkill]] | None = None,
        ) -> None:
        self.lock = threading.Lock()
        self.api = {}
        self.canonical_vocabulary_adapter = SkillVocabularyAdapter()
        self.identity_vocabulary_adapter = SkillVocabularyAdapter()
        self._existing_skills_type = {
            skill_type.spec.name: skill_type
            for skill_type in (valid_skill_type or [])
        }
        with self.lock:
            self.running_skill_per_node = {}
            self.skill_states: Dict[SkillStateRef, RunningSkillState] = {}
            self.next_state_ref = 1
            self.pending_control_statuses = {}
            self.has_skill_waiting = False

    def _store_state(self, state: RunningSkillState) -> SkillStateRef:
        state_ref = SkillStateRef(self.next_state_ref)
        self.next_state_ref += 1
        self.skill_states[state_ref] = deepcopy(state)
        return state_ref

    def create_initial_state(
        self,
        execution_context: SkillExecutionContext,
    ) -> SkillStateRef:
        with self.lock:
            return self._store_state(RunningSkillState(0, 0, execution_context))

    def get_suspended_answer(
        self,
        state_ref: SkillStateRef,
    ) -> Optional[ValidAgentAnswer]:
        with self.lock:
            state = self.skill_states.get(state_ref)
            if state is None:
                raise RuntimeError(f"Unknown skill state reference {state_ref.value}")
            suspended_work = state.get_suspended_work()
            if suspended_work is None:
                return None
            return deepcopy(suspended_work.answer)

    def resume(
        self,
        node_id: int,
        state_ref: SkillStateRef,
        execution_context: SkillExecutionContext,
    ) -> None:
        with self.lock:
            parent_state = self.skill_states.get(state_ref)
            if parent_state is None:
                raise RuntimeError(f"Unknown skill state reference {state_ref.value}")
            suspended_work = parent_state.get_suspended_work()
            if suspended_work is None:
                raise RuntimeError("Cannot resume a skill state without suspended work")
            if node_id in self.running_skill_per_node:
                raise RuntimeError(f"Node {node_id} already owns a running skill state")

            suspended = suspended_work.answer
            running_state = parent_state.create_child(
                parent_state.source_node_id,
                suspended.agent_step_id,
                execution_context,
            )
            self.running_skill_per_node[node_id] = running_state
            empty = ValidAgentAnswer(
                source_node_id=parent_state.source_node_id,
                agent_step_id=suspended.agent_step_id,
                say="",
                calls=[],
            )
            _, handled = self._resolve_pending_answer(running_state, empty)
            if not handled:
                raise RuntimeError("Suspended work could not be resumed")
            self.has_skill_waiting = True

    def build_api(self, executor: SkillAPIProvider):
        self.canonical_vocabulary_adapter = SkillVocabularyAdapter(
            executor.get_skill_vocabulary_translator()
        )
        tools_with_real_names = executor.get_tools_with_real_names()
        real_tool_names = [real_name for _, real_name in tools_with_real_names]
        used_tool = set()
        self.api = {
            CancelCurrentActionSkill.spec.name: CancelCurrentActionSkill,
        }
        for skill_name, SkillCLS in self._existing_skills_type.items():
            if skill_name in self.api:
                raise RuntimeError(f"Skill name {skill_name} is reserved")
            for t in SkillCLS.spec.required_tools:
                if t not in real_tool_names:
                    raise RuntimeError(
                        f"Skill {SkillCLS.spec.name} requires tool {t} but it's "
                        f"not in tool api from scenario ({real_tool_names})"
                    )
                used_tool.add(t)
            self.api[skill_name] = SkillCLS
        for tool_dict, real_name in tools_with_real_names:
            if real_name not in used_tool:
                if tool_dict['name'] in self.api:
                    raise RuntimeError(f"Tool name {tool_dict['name']} is reserved")
                self.api[tool_dict['name']] = build_mono_tool_skill_class(tool_dict)

    def register(
            self,
            registrations: Dict[int, SkillRegistration],
        ) -> None:
        """
        Register agent answer inside each active node RunningState.

        These answer will be launched to executor or send again to agents via the tick in the main thread.
        """
        with self.lock:
            for node_id, registration in registrations.items():
                answer = registration.answer
                # print(f"Receiving {node_id} with {answer.to_dict()}")
                if not answer.is_valid():
                    raise RuntimeError("An invalid answer has reached the skill registration")
                # Creating or Getting RUNNING STATE ======
                if self.running_skill_per_node.get(node_id,None) is not None:
                    raise RuntimeError(f"Node {node_id} trying to register a running state "
                                       "whereas there already exist one.")

                parent_state = self.skill_states.get(registration.state_ref)
                if parent_state is None:
                    raise RuntimeError(
                        f"Unknown skill state reference {registration.state_ref.value}"
                    )
                running_state = parent_state.create_child(
                    answer.source_node_id,
                    answer.agent_step_id,
                    registration.execution_context,
                )
                self.running_skill_per_node[node_id] = running_state

                if not isinstance(answer, ValidAgentAnswer):
                    raise TypeError("Only ValidAgentAnswer can reach skill registration")
                if answer.get_say() != "" and answer.get_action():
                    raise ValueError("An answer cannot contain both say and tool calls")

                if isinstance(running_state.interruption, AwaitingDeferredAnswer):
                    running_state.capture_before_pending_input(answer)
                    self.has_skill_waiting = True
                    continue

                control_statuses, handled = self._resolve_pending_answer(
                    running_state,
                    answer,
                )
                if not handled:
                    control_statuses.extend(
                        self._register_answer_in_state(running_state, answer)
                    )

                if running_state.failure_reason is not None:
                    control_statuses.append(RobotToolStatus(
                        running_state.failure_reason[0],
                        running_state.failure_reason[1],
                        False,
                        ToolErrorFlag.BAD_CALL,
                    ))
                    running_state.failure_reason = None
                if control_statuses:
                    self.pending_control_statuses[node_id] = control_statuses

                self.has_skill_waiting = True

    def _resolve_pending_answer(
        self,
        running_state: RunningSkillState,
        answer: ValidAgentAnswer,
    ) -> Tuple[List[RobotToolStatus], bool]:
        suspended = running_state.get_suspended_work()
        if suspended is None:
            return [], False

        pending = suspended.answer

        pending_calls = pending.get_action()
        if pending_calls:
            cancel_calls = [
                call for call in answer.get_action()
                if self.api.get(call.name) is CancelCurrentActionSkill
            ]
            if answer.is_empty():
                running_state.set_suspended_work(None)
                running_state.resumed_work = deepcopy(suspended)
                running_state.agent_step_id = pending.agent_step_id
                running_state.executed_answer = deepcopy(pending)
                if isinstance(suspended, AgentSuspendedWork):
                    resumed = ValidAgentAnswer(
                        source_node_id=answer.source_node_id,
                        agent_step_id=pending.agent_step_id,
                        say="",
                        calls=deepcopy(pending_calls),
                    )
                    self._register_answer_in_state(running_state, resumed)
                    running_state.executed_answer = deepcopy(pending)
                return [], True

            if cancel_calls and len(cancel_calls) == len(answer.get_action()):
                pending_by_robot = {
                    call.target_robot_name: call for call in pending_calls
                }
                statuses = []
                for cancel_call in cancel_calls:
                    pending_call = pending_by_robot.pop(
                        cancel_call.target_robot_name,
                        None,
                    )
                    if pending_call is None:
                        statuses.append(RobotToolStatus(
                            cancel_call.target_robot_name,
                            "No action is currently running on this robot.",
                            False,
                            ToolErrorFlag.BAD_CALL,
                        ))
                        continue
                    statuses.append(RobotToolStatus(
                        cancel_call.target_robot_name,
                        f"{pending_call.name} was cancelled.",
                        True,
                        ToolErrorFlag.NONE,
                    ))
                if not pending_by_robot:
                    running_state.set_suspended_work(None)
                else:
                    work_type = type(suspended)
                    running_state.set_suspended_work(work_type(
                        answer=ValidAgentAnswer(
                            source_node_id=pending.source_node_id,
                            agent_step_id=pending.agent_step_id,
                            say="",
                            calls=list(pending_by_robot.values()),
                        ),
                        origin_stage_id=suspended.origin_stage_id,
                    ))
                if isinstance(suspended, SkillTickSuspendedWork):
                    cancelled_robots = {
                        call.target_robot_name for call in cancel_calls
                    }
                    running_state.running_skills = [
                        skill_state
                        for skill_state in running_state.running_skills
                        if skill_state.robot_name not in cancelled_robots
                    ]
                running_state.executed_answer = deepcopy(answer)
                return statuses, True

            if answer.get_action():
                running_state.executed_answer = deepcopy(answer)
                return [
                    RobotToolStatus(
                        call.target_robot_name,
                        (
                            f"Impossible to launch {call.name} because "
                            "another action is suspended on this robot. Use "
                            f"{call.target_robot_name}.cancel_current_action() first."
                        ),
                        False,
                        ToolErrorFlag.BAD_CALL,
                    )
                    for call in answer.get_action()
                ], True

            # A say is evaluated normally while the tool remains suspended.
            return [], False

        # A pending say is replaced by any non-empty response.
        running_state.set_suspended_work(None)
        if answer.is_empty():
            running_state.resumed_work = deepcopy(suspended)
            resumed = ValidAgentAnswer(
                source_node_id=answer.source_node_id,
                agent_step_id=pending.agent_step_id,
                say=pending.get_say(),
                calls=[],
            )
            running_state.agent_step_id = pending.agent_step_id
            running_state.executed_answer = deepcopy(pending)
            self._register_answer_in_state(running_state, resumed)
            running_state.executed_answer = deepcopy(pending)
            return [], True
        return [], False

    def _register_answer_in_state(
        self,
        running_state: RunningSkillState,
        answer: ValidAgentAnswer,
    ) -> List[RobotToolStatus]:
        calls = answer.get_action()
        running_state.executed_answer = deepcopy(answer)
        running_state.add_user_answer(
            answer.get_say(),
            requires_execution=not calls,
        )
        cancel_robots = {
            call.target_robot_name
            for call in calls
            if self.api.get(call.name) is CancelCurrentActionSkill
        }
        regular_robots = {
            call.target_robot_name
            for call in calls
            if self.api.get(call.name) is not CancelCurrentActionSkill
        }
        conflicting_robots = cancel_robots & regular_robots
        control_statuses = [
            RobotToolStatus(
                robot_name,
                "Cannot cancel and start an action on the same robot in one answer.",
                False,
                ToolErrorFlag.BAD_CALL,
            )
            for robot_name in conflicting_robots
        ]

        for robot_name in cancel_robots - conflicting_robots:
            control_statuses.append(running_state.cancel_skill(robot_name))
        if (
            cancel_robots
            and not running_state.running_skills
            and running_state.get_suspended_work() is None
        ):
            running_state.executed_answer = deepcopy(answer)

        for c in calls:
            if c.name not in self.api:
                control_statuses.append(
                    RobotToolStatus(
                        c.target_robot_name,
                        f"{c.name} is not a known tool.",
                        False,
                        ToolErrorFlag.BAD_CALL,
                        mess_is_public=True,
                    )
                )
                continue
            if self.api[c.name] is CancelCurrentActionSkill:
                continue
            if c.target_robot_name in conflicting_robots:
                continue
            vocabulary_adapter = (
                self.canonical_vocabulary_adapter
                if c.name in self._existing_skills_type
                else self.identity_vocabulary_adapter
            )
            try:
                arguments = vocabulary_adapter.to_runtime_arguments(c.arguments)
            except ValueError as error:
                control_statuses.append(
                    RobotToolStatus(
                        c.target_robot_name,
                        str(error),
                        False,
                        ToolErrorFlag.BAD_CALL,
                        mess_is_public=True,
                    )
                )
                continue
            skill_instance = self.api[c.name](arguments)

            running_state.add_new_skill(
                c.target_robot_name,
                skill_instance,
                deepcopy(c),
                vocabulary_adapter,
            )

        return control_statuses


    
    def tick(self, status_per_node : Dict[int, ToolStatus]) -> Tick:
        """
        Tick all skills running for all nodes.
        """
        results: Dict[int, SkillNodeResult] = {}

        with self.lock:
            if not self.has_skill_waiting and len(status_per_node) == 0:
                return Tick({})
            
            self.has_skill_waiting = False

            for node_id, running_state in self.running_skill_per_node.items():
                environment_source_node_id = running_state.source_node_id

                control_statuses = self.pending_control_statuses.pop(node_id, None)
                if control_statuses is not None:
                    status = ToolStatus(
                        robots_status=control_statuses,
                        stage_id=running_state.execution_context.stage_id,
                        attributes=deepcopy(running_state.execution_context.attributes),
                        error_descriptions=[""] * len(control_statuses),
                        stage_success=StageSuccess.ONGOING,
                        tool_calls=running_state.execution_context.tool_calls,
                        forgiven_tool_calls=(
                            running_state.execution_context.forgiven_tool_calls
                        ),
                    )
                    self._update_robot_statuses(status.attributes, running_state)
                    state_ref = self._store_state(running_state)
                    event = SkillStatusEvent(
                        status=status,
                        state_ref=state_ref,
                        executed_answer=deepcopy(running_state.executed_answer),
                        has_suspended_work=running_state.get_suspended_work() is not None,
                    )
                    results[node_id] = SkillStatusResult(
                        environment_source_node_id=environment_source_node_id,
                        status_event=event,
                    )
                    continue

                # tool_status will be None if there is no status
                tool_status = status_per_node.get(node_id,None)
                if tool_status is None and isinstance(
                    running_state.interruption,
                    DeferredInputReady,
                ):
                    pending_input = running_state.pop_pending_input()
                    self._update_robot_statuses(
                        pending_input.attributes,
                        running_state,
                    )
                    state_ref = self._store_state(running_state)
                    transition = DeferredInputTransition(
                        stage_input=pending_input.stage_input,
                        attributes=pending_input.attributes,
                        tool_calls=pending_input.tool_calls,
                        forgiven_tool_calls=pending_input.forgiven_tool_calls,
                        state_ref=state_ref,
                    )
                    results[node_id] = SkillDeferredInputResult(
                        environment_source_node_id=environment_source_node_id,
                        deferred_input=transition,
                    )
                    continue
                if tool_status is None:
                    # Try to see if there is a tool waiting for a start
                    if not running_state.has_waiting_for_tick():
                        continue                    
                
                # We reach that point either because:
                # - All skills running have their tool finished and need update
                # - One agent answer for the running state has been received
                # and so we need to tick all skills again.
                # print(f"ticking {node_id}")
                if tool_status is not None:
                    running_state.source_node_id = node_id
                    if running_state.resumed_work is not None:
                        if tool_status.stage_success == StageSuccess.FAILED:
                            running_state.set_suspended_work(
                                running_state.resumed_work
                            )
                            running_state.resumed_work = None
                    if (
                        running_state.say_sent
                        and tool_status.stage_success == StageSuccess.ONGOING
                        and tool_status.next_input is None
                    ):
                        suspended = running_state.get_suspended_work()
                        if suspended is not None and suspended.answer.get_action():
                            pending = suspended.answer
                            running_state.set_suspended_work(None)
                            running_state.resumed_work = deepcopy(suspended)
                            running_state.agent_step_id = pending.agent_step_id
                            running_state.say = ""
                            running_state.say_pending = False
                            running_state.say_sent = False
                            running_state.executed_answer = deepcopy(pending)
                            if isinstance(suspended, AgentSuspendedWork):
                                resumed = ValidAgentAnswer(
                                    source_node_id=node_id,
                                    agent_step_id=pending.agent_step_id,
                                    say="",
                                    calls=deepcopy(pending.get_action()),
                                )
                                self._register_answer_in_state(running_state, resumed)
                                running_state.executed_answer = deepcopy(pending)
                            tool_status = None
                        elif any(
                            skill_state.wait_for_tick()
                            for skill_state in running_state.running_skills
                        ):
                            running_state.say = ""
                            running_state.say_pending = False
                            running_state.say_sent = False
                            tool_status = None
                req, ended_status = running_state.tick(tool_status)
                # print(req, status)
                running_state.clean_finished_skill()
                if req is not None and ended_status is not None:
                    raise RuntimeError(
                        "A skill transition cannot execute work and end a node together"
                    )
                if ended_status is not None:
                    running_state.resumed_work = None
                if req is not None:
                    results[node_id] = SkillExecutionResult(
                        environment_source_node_id=environment_source_node_id,
                        request=req,
                        executed_answer=deepcopy(running_state.executed_answer),
                    )
                if ended_status is not None:
                    self._update_robot_statuses(
                        ended_status.attributes,
                        running_state,
                    )
                    state_ref = self._store_state(running_state)
                    status_event = SkillStatusEvent(
                        status=ended_status,
                        state_ref=state_ref,
                        executed_answer=deepcopy(running_state.executed_answer),
                        has_suspended_work=running_state.get_suspended_work() is not None,
                    )
                    results[node_id] = SkillStatusResult(
                        environment_source_node_id=environment_source_node_id,
                        status_event=status_event,
                    )

            for node_id, result in results.items():
                if not isinstance(result, SkillExecutionResult):
                    self.running_skill_per_node.pop(node_id, None)

        return Tick(results)
    
    def get_api(self) -> List[Dict]:
        return [skill.spec.to_dict() for skill in self.api.values()]

    def initialize_robot_statuses(
        self,
        attributes: Dict[str, Any],
        stage_id: int = 0,
    ) -> SkillStateRef:
        known_robots = attributes.get("known_robots")
        if known_robots is None:
            raise RuntimeError("Cannot initialize robot statuses without known_robots")
        attributes["robot_statuses"] = {
            robot_name: "FREE"
            for robot_name in known_robots
        }
        return self.create_initial_state(SkillExecutionContext(
            stage_id=stage_id,
            attributes=deepcopy(attributes),
        ))

    def reset_states(self) -> None:
        with self.lock:
            self.running_skill_per_node.clear()
            self.skill_states.clear()
            self.pending_control_statuses.clear()
            self.next_state_ref = 1
            self.has_skill_waiting = False

    def _update_robot_statuses(
        self,
        attributes: Dict,
        running_state: RunningSkillState,
    ) -> None:
        known_robots = attributes.get("known_robots")
        if known_robots is None:
            raise RuntimeError(
                "Cannot expose robot statuses without known_robots"
            )
        robot_statuses = {
            robot_name: "FREE"
            for robot_name in known_robots
        }

        for skill_state in running_state.running_skills:
            robot_statuses[skill_state.robot_name] = (
                f"RUNNING {skill_state.public_call.to_function_string()}"
            )

        suspended_work = running_state.get_suspended_work()
        if suspended_work is not None and suspended_work.answer.get_action():
            for call in suspended_work.answer.get_action():
                robot_statuses[call.target_robot_name] = (
                    f"RUNNING {call.to_function_string()}"
                )

        attributes["robot_statuses"] = robot_statuses
