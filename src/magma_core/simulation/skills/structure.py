from __future__ import annotations
from dataclasses import dataclass
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from magma_core.domain import Call, ValidExecutionReq
from magma_core.simulation.data_structures import (
    EmptyInstruction,
    RobotToolStatus,
    StageSuccess,
    ToolErrorFlag,
    ToolStatus,
    StageInput,
)
from magma_core.simulation.agents import ValidAgentAnswer
from magma_core.simulation.skills.adapter import SkillVocabularyAdapter
from magma_core.simulation.skills.skill import (
    BaseSkill, SkillStatus, SkillTick, MessTick, CallTick,
)


@dataclass(frozen=True)
class SkillExecutionContext:
    """Runtime context used only when a skill ends before calling a tool."""

    stage_id: int
    attributes: Dict[str, Any]
    tool_calls: int = 0
    forgiven_tool_calls: int = 0
    flag_answer_to_user: bool = False


@dataclass(frozen=True)
class SkillStateRef:
    """Opaque reference to an immutable branch-local skill snapshot."""

    value: int


@dataclass(frozen=True)
class SkillRegistration:
    """Everything required to apply one answer to a branch skill state."""

    answer: ValidAgentAnswer
    execution_context: SkillExecutionContext
    state_ref: SkillStateRef


@dataclass(frozen=True)
class DeferredInputTransition:
    """Publish a delayed stage input without executing the captured answer."""

    stage_input: StageInput
    attributes: Dict[str, Any]
    tool_calls: int
    forgiven_tool_calls: int
    state_ref: SkillStateRef


@dataclass(frozen=True)
class SkillStatusEvent:
    """Internal result passed from the skill layer to graph termination."""

    status: ToolStatus
    state_ref: SkillStateRef
    executed_answer: Optional[ValidAgentAnswer] = None
    has_suspended_work: bool = False


@dataclass(frozen=True)
class PendingInput:
    stage_input: StageInput
    attributes: Dict[str, Any]
    tool_calls: int
    forgiven_tool_calls: int


@dataclass(frozen=True)
class AgentSuspendedWork:
    """Agent answer captured before publishing a delayed instruction."""

    answer: ValidAgentAnswer
    origin_stage_id: int


@dataclass(frozen=True)
class SkillTickSuspendedWork:
    """Internal skill call captured before publishing a delayed instruction."""

    answer: ValidAgentAnswer
    origin_stage_id: int


SuspendedWork = AgentSuspendedWork | SkillTickSuspendedWork


@dataclass(frozen=True)
class AwaitingDeferredAnswer:
    """The tool feedback is visible and the delayed input is not published yet."""

    pending_input: PendingInput


@dataclass(frozen=True)
class DeferredInputReady:
    """The pre-interruption answer is captured and the input can be published."""

    pending_input: PendingInput
    suspended_work: Optional[SuspendedWork]


@dataclass(frozen=True)
class SuspendedWorkState:
    """A delayed input was published and its captured work awaits resolution."""

    suspended_work: SuspendedWork


InterruptionState = (
    AwaitingDeferredAnswer
    | DeferredInputReady
    | SuspendedWorkState
)


@dataclass(frozen=True)
class SkillExecutionResult:
    """Physical work to send to the executor."""

    environment_source_node_id: int
    request: ValidExecutionReq
    executed_answer: Optional[ValidAgentAnswer] = None


@dataclass(frozen=True)
class SkillStatusResult:
    """Feedback or completion to send to graph termination."""

    environment_source_node_id: int
    status_event: SkillStatusEvent


@dataclass(frozen=True)
class SkillDeferredInputResult:
    """Delayed instruction to publish through the graph input pipeline."""

    environment_source_node_id: int
    deferred_input: DeferredInputTransition


SkillNodeResult = SkillExecutionResult | SkillStatusResult | SkillDeferredInputResult


class RunningSkillCallState:
    skill: BaseSkill
    robot_name: str
    public_call: Call

    status : SkillStatus
    # The saved tick is set when we tick all skill from the node and 
    # that one of them is sending an update to agents.
    # To synchronize, we keep the tick and wait for the agent answer to came back
    # before sending the tick to execution.
    saved_tick : Optional[SkillTick]
    vocabulary_adapter: SkillVocabularyAdapter

    def __init__(
            self,
            skill : BaseSkill,
            robot_name : str,
            public_call: Call,
            vocabulary_adapter: SkillVocabularyAdapter,
        ) -> None:
        self.status = SkillStatus.WAITING_START
        self.saved_tick = None
        self.skill = skill
        self.robot_name = robot_name
        self.public_call = public_call
        self.vocabulary_adapter = vocabulary_adapter

    def wait_for_tick(self) -> bool:
        return self.status in {SkillStatus.WAITING_START, SkillStatus.WAITING_TICK}
    
    def empty_tick(self) -> SkillTick:
        if self.status == SkillStatus.WAITING_TICK:
            if self.saved_tick is None:
                raise RuntimeError("Skill state is WAITING_TICK without a saved tick")
            out = self.saved_tick
            self.saved_tick = None
            return out

        if self.status != SkillStatus.WAITING_START:
            raise RuntimeError(f"Cannot tick a skill without tool status from state {self.status}")
        
        return self.skill.start()

    def tick(self, status: RobotToolStatus) -> SkillTick:        
        return self.skill.tick(status)

class RunningSkillState:
    """
    Stock reference to skill instances associated with a node
    The node id is not directly Stocked here.
    """
    source_node_id: int
    agent_step_id: int

    running_skills: List[RunningSkillCallState]
    say : str

    say_sent : bool

    # Set only in case of failure in this node
    failure_reason : Optional[Tuple[str,str]]

    def __init__(
            self,
            source_node_id : int,
            agent_step_id : int,
            execution_context: SkillExecutionContext,
        ) -> None:
        self.agent_step_id = agent_step_id
        self.source_node_id = source_node_id
        self.execution_context = execution_context
        self.running_skills = []
        self.say = ""
        self.say_pending = False
        self.say_sent = False
        self.activated = True
        self.failure_reason = None
        self.interruption: Optional[InterruptionState] = None
        self.executed_answer: Optional[ValidAgentAnswer] = None
        # Execution provenance exists only while resumed work is physically
        # running. A failed result restores it into ``interruption``.
        self.resumed_work: Optional[SuspendedWork] = None

    def add_user_answer(self, say : str, *, requires_execution: bool):
        self.say = say
        self.say_pending = requires_execution

    def add_new_skill(
            self,
            robot_name : str,
            skill : BaseSkill,
            public_call: Call,
            vocabulary_adapter: SkillVocabularyAdapter,
        ):
        if self.say != "":
            raise RuntimeError(f"Trying to add a call {skill.spec.name} whereas say is already set: {self.say}")
        for sk in self.running_skills:
            if robot_name == sk.robot_name:
                self.failure_reason=robot_name, (f"Impossible to launch {skill.spec.name} on robot {sk.robot_name} "
                                     f" because {sk.skill.spec.name} is already running on it. You must "
                                     f"use {robot_name}.cancel_current_action() first."
                                    )
                # print("failure reason : ", self.failure_reason)
                break
        else:
            self.running_skills.append(
                RunningSkillCallState(
                    skill,
                    robot_name,
                    public_call,
                    vocabulary_adapter,
                ))

    def create_child(
            self,
            source_node_id : int,
            agent_step_id : int,
            execution_context: SkillExecutionContext,
        ) -> RunningSkillState:
        out = RunningSkillState(
            source_node_id,
            agent_step_id,
            execution_context,
        )
        for skill_state in self.running_skills:
            copied_state = RunningSkillCallState(
                deepcopy(skill_state.skill),
                skill_state.robot_name,
                deepcopy(skill_state.public_call),
                skill_state.vocabulary_adapter,
            )
            copied_state.status = skill_state.status
            copied_state.saved_tick = deepcopy(skill_state.saved_tick)
            out.running_skills.append(copied_state)
        out.interruption = deepcopy(self.interruption)
        out.executed_answer = deepcopy(self.executed_answer)
        return out

    def capture_before_pending_input(self, answer: ValidAgentAnswer) -> None:
        if not isinstance(self.interruption, AwaitingDeferredAnswer):
            raise RuntimeError("Cannot capture an answer without a pending input")
        suspended_work = None
        if not answer.is_empty():
            suspended_work = AgentSuspendedWork(
                answer=deepcopy(answer),
                origin_stage_id=self.execution_context.stage_id,
            )
        self.interruption = DeferredInputReady(
            pending_input=self.interruption.pending_input,
            suspended_work=suspended_work,
        )

    def pop_pending_input(self) -> PendingInput:
        if not isinstance(self.interruption, DeferredInputReady):
            raise RuntimeError("No delayed input is waiting")
        ready = self.interruption
        self.interruption = (
            SuspendedWorkState(ready.suspended_work)
            if ready.suspended_work is not None
            else None
        )
        return ready.pending_input

    def get_suspended_work(self) -> Optional[SuspendedWork]:
        if isinstance(self.interruption, SuspendedWorkState):
            return self.interruption.suspended_work
        return None

    def set_suspended_work(self, work: Optional[SuspendedWork]) -> None:
        self.interruption = SuspendedWorkState(work) if work is not None else None
    
    def clean_finished_skill(self):
        updated_running = [
            skill_state for skill_state in self.running_skills 
            if skill_state.status != SkillStatus.FINISH
        ]
        self.running_skills = updated_running

    def cancel_skill(self, robot_name: str) -> RobotToolStatus:
        for index, skill_state in enumerate(self.running_skills):
            if skill_state.robot_name != robot_name:
                continue

            skill_name = skill_state.skill.spec.name
            self.running_skills.pop(index)
            return RobotToolStatus(
                robot_name,
                f"{skill_name} was cancelled.",
                True,
                ToolErrorFlag.NONE,
            )

        return RobotToolStatus(
            robot_name,
            "No action is currently running on this robot.",
            False,
            ToolErrorFlag.BAD_CALL,
        )


    
    def tick(
        self,
        precedent_tool_status: Optional[ToolStatus],
    ) -> Tuple[Optional[ValidExecutionReq], Optional[ToolStatus]]:
        if not self.activated:
            raise RuntimeError("Ticking a finished State")

        if self.say_pending:
            if precedent_tool_status is None:
                if self.say_sent:
                    raise RuntimeError(
                        "An answer waiting for its result was ticked without a status"
                    )
                self.say_sent = True
                return ValidExecutionReq(
                    self.source_node_id,
                    self.agent_step_id,
                    [],
                    self.say,
                ), None

            self.say = ""
            self.say_pending = False
            self.say_sent = False
            if (
                precedent_tool_status.stage_success == StageSuccess.FINISH
                and precedent_tool_status.next_input is None
            ):
                self.interruption = None
                self.running_skills = []
            self.activated = False
            return None, precedent_tool_status

        has_delayed_input = (
            precedent_tool_status is not None
            and precedent_tool_status.stage_success == StageSuccess.FINISH
            and precedent_tool_status.next_input is not None
            and not self.execution_context.flag_answer_to_user
            and not isinstance(
                precedent_tool_status.next_input.instruction,
                EmptyInstruction,
            )
        )
        if has_delayed_input:
            if self.interruption is not None:
                raise RuntimeError("A delayed stage input is already waiting")
            captured_input = deepcopy(precedent_tool_status.next_input)
            self.interruption = AwaitingDeferredAnswer(
                PendingInput(
                    stage_input=captured_input,
                    attributes=deepcopy(precedent_tool_status.attributes),
                    tool_calls=precedent_tool_status.tool_calls,
                    forgiven_tool_calls=precedent_tool_status.forgiven_tool_calls,
                )
            )
            skill_visible_status = deepcopy(precedent_tool_status)
            skill_visible_status.next_input = StageInput(
                instruction=EmptyInstruction(),
                flag_answer_to_user=captured_input.flag_answer_to_user,
                linked_to_prev=captured_input.linked_to_prev,
            )
        else:
            skill_visible_status = precedent_tool_status

        ticks : List[SkillTick] = []

        for skill_state in self.running_skills:
            if skill_visible_status is not None:
                skill_return_status = skill_visible_status.get_robot_status(
                    skill_state.robot_name
                )

                if skill_return_status is not None:          
                    skill_return_status = skill_state.vocabulary_adapter.to_runtime_status(
                        skill_return_status
                    )
                    tick = skill_state.tick(skill_return_status)
                    ticks.append(tick)
                    continue
        
                    
            if skill_state.wait_for_tick():
                tick = skill_state.empty_tick()
                ticks.append(tick)
            else:
                raise RuntimeError(f"A running skill {skill_state.status} got an empty status")
        
        must_return_to_agent = (
            any(isinstance(tick, MessTick) for tick in ticks)
            or self.failure_reason is not None
        )
        added = False
        call_list : List[Call] = []
        status : List[RobotToolStatus] = []

        execution_req = None
        tool_status = None

        for idx, tick in enumerate(ticks):
            robot_name = self.running_skills[idx].robot_name
            if isinstance(tick,CallTick):
                if must_return_to_agent:
                    if self.running_skills[idx].saved_tick != None:
                        raise RuntimeError("Trying to save a tick whereas already one was here")
                    self.running_skills[idx].saved_tick = tick
                    self.running_skills[idx].status = SkillStatus.WAITING_TICK
                else:
                    self.running_skills[idx].status = SkillStatus.RUNNING
                    call_list.append(
                        self.running_skills[idx].vocabulary_adapter.to_public_call(
                            tick,
                            robot_name,
                        )
                    )
            else:
                self.running_skills[idx].status = SkillStatus.FINISH
                m : str = tick.message
                if self.failure_reason is not None and self.failure_reason[0] == robot_name:
                    added=True
                    m += "\n " + self.failure_reason[1]
                m = self.running_skills[idx].vocabulary_adapter.to_public_message(m)

                status.append(RobotToolStatus(
                    robot_name,
                    m,
                    tick.result,
                    tick.error_flag,
                ))

        if self.failure_reason is not None and not added:
            status.append(
                RobotToolStatus(
                    self.failure_reason[0],
                    self.failure_reason[1],
                    False,
                    ToolErrorFlag.BAD_CALL
                )
            )

        has_call = len(call_list) > 0
        has_status = len(status) > 0

        if has_call and has_status:
            raise RuntimeError("A node is trying to call and return a status to agent at the same time")

        if has_call:
            if (
                isinstance(self.interruption, AwaitingDeferredAnswer)
                and precedent_tool_status is not None
            ):
                deferred_calls: List[Call] = []
                for idx, tick in enumerate(ticks):
                    if not isinstance(tick, CallTick):
                        continue
                    self.running_skills[idx].saved_tick = tick
                    self.running_skills[idx].status = SkillStatus.WAITING_TICK
                    deferred_calls.append(self.running_skills[idx].public_call)
                suspended_answer = ValidAgentAnswer(
                    source_node_id=self.source_node_id,
                    agent_step_id=self.agent_step_id,
                    say="",
                    calls=deepcopy(deferred_calls),
                )
                self.set_suspended_work(SkillTickSuspendedWork(
                    answer=suspended_answer,
                    origin_stage_id=self.execution_context.stage_id,
                ))
                self.activated = False
                tool_status = deepcopy(precedent_tool_status)
            else:
                execution_req = ValidExecutionReq(
                    self.source_node_id,
                    self.agent_step_id,
                    calls=call_list,
                    say=""
                )
                if self.executed_answer is None:
                    self.executed_answer = ValidAgentAnswer(
                        source_node_id=self.source_node_id,
                        agent_step_id=self.agent_step_id,
                        say="",
                        calls=[
                            deepcopy(skill_state.public_call)
                            for skill_state in self.running_skills
                            if skill_state.status == SkillStatus.RUNNING
                        ],
                    )
        
        if has_status:
            if skill_visible_status:
                tool_status = deepcopy(skill_visible_status)
                tool_status.robots_status = status
            else:
                # If there is no status here that's mean that it's an error
                # from the skill
                tool_status = ToolStatus(
                    robots_status=status,
                    stage_id=self.execution_context.stage_id,
                    attributes=deepcopy(self.execution_context.attributes),
                    error_descriptions=[""] * len(status),
                    tool_calls=self.execution_context.tool_calls,
                    forgiven_tool_calls=self.execution_context.forgiven_tool_calls,
                )
            self.activated = False

        return execution_req, tool_status

    def has_waiting_for_tick(self) -> bool:
        return isinstance(
            self.interruption,
            (AwaitingDeferredAnswer, DeferredInputReady),
        ) or (self.say_pending and not self.say_sent) or any(
            state.wait_for_tick()
            for state in self.running_skills
        )


@dataclass(frozen=True)
class Tick:
    """
    The Tick structure stock important ticking data:
    - valid_request for executor
    - ended_status for node end manager
    """
    results: Dict[int, SkillNodeResult]

    def has_execution(self) -> bool:
        return any(
            isinstance(result, SkillExecutionResult)
            for result in self.results.values()
        )
    
    def has_ended_status(self) -> bool:
        return any(
            isinstance(result, SkillStatusResult)
            for result in self.results.values()
        )
