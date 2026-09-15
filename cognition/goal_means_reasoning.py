"""Goal-means coherence for practical conversational decisions.

This module does not solve a user's task with brittle domain rules.  It
recognizes when a turn asks for an action choice and supplies a compact
reasoning protocol to the response model: preserve the primary outcome,
check material prerequisites, then consider secondary preferences.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List


_DECISION_PATTERNS = (
    r"\bshould\s+(?:i|we)\b",
    r"\bhow\s+(?:can|could|should)\s+(?:i|we)\b",
    r"\bdo\s+i\b.{0,100}\bor\b",
    r"\bwhich\s+(?:option|way|route|method|one)\b",
    r"\b(?:dois|devrais)[- ]?je\b",
    r"\bcomment\s+(?:puis|pourrais|devrais)[- ]?je\b",
    r"\best[- ]?ce\s+que\b",
    r"\bquelle?\s+(?:option|façon|manière|méthode)\b",
)


def is_practical_decision(text: str) -> bool:
    """Return whether a turn asks for a concrete choice or course of action."""
    normalized = " ".join(str(text or "").strip().casefold().split())
    if len(normalized) < 8:
        return False
    return any(re.search(pattern, normalized) for pattern in _DECISION_PATTERNS)


def goal_means_contract(text: str) -> str:
    """Build a domain-independent reasoning contract for practical choices."""
    if not is_practical_decision(text):
        return ""
    return """━━ GOAL-MEANS COHERENCE CHECK ━━
This is a practical action or choice question. Reason in this order before answering:
1. Identify the human's primary intended outcome. Preserve it unchanged.
2. Treat distance, comfort, weather, speed, cost, and similar details as secondary unless they determine feasibility.
3. Identify the agents, objects, resources, locations, and prior conditions that the primary action requires.
4. Simulate each proposed option concretely: after choosing it, are all indispensable prerequisites present where and when the primary action must happen?
   Track temporal order and initial locations. Do not place one agent in two incompatible locations at the same time or invent a helper or remote trigger.
5. Check the real agent's physical capability, including reaction time, acceleration, anatomy, route constraints, environmental limits, and safety. A computable requirement can still be physically impossible.
6. Eliminate any option that prevents the primary outcome, even if that option is attractive on a secondary criterion.
7. Recommend a feasible option directly first. Discuss preferences or richer interpretation only after feasibility is established.

Do not silently change the goal, invent unmentioned equipment or workarounds, or turn a straightforward logistical constraint into a metaphor. If the human corrects a missed prerequisite, revise the original model from that correction instead of defending the rejected option."""


@dataclass
class OptionAssessment:
    name: str
    feasible: bool
    reason: str
    agent_at_execution_location: bool | None = None
    patient_at_execution_location: bool | None = None
    moves_patient_or_theme: bool | None = None
    required_resources_available: bool | None = None
    agent_capability_sufficient: bool | None = None


@dataclass
class GoalMeansAnalysis:
    primary_goal: str
    semantic_roles: dict = field(default_factory=dict)
    patient_must_move_to_execution_location: bool | None = None
    required_conditions: List[str] = field(default_factory=list)
    options: List[OptionAssessment] = field(default_factory=list)
    recommended_option: str = ""
    decisive_reason: str = ""

    def prompt_fragment(self) -> str:
        lines = [
            "━━ STRUCTURED GOAL-MEANS ANALYSIS ━━",
            f"Primary intended outcome: {self.primary_goal}",
        ]
        if self.required_conditions:
            lines.append("Indispensable conditions: " + "; ".join(self.required_conditions))
        if self.semantic_roles:
            lines.append(
                "Material roles: "
                + "; ".join(
                    f"{key}={value}" for key, value in self.semantic_roles.items()
                )
            )
        if self.patient_must_move_to_execution_location is not None:
            lines.append(
                "Acted-on object must move to the execution location: "
                + ("YES" if self.patient_must_move_to_execution_location else "NO")
            )
        for option in self.options:
            status = "FEASIBLE" if option.feasible else "INFEASIBLE"
            lines.append(f"Option {option.name}: {status} - {option.reason}")
        if self.recommended_option:
            lines.append(f"Recommendation: {self.recommended_option}")
        if self.decisive_reason:
            lines.append(f"Decisive reason: {self.decisive_reason}")
        lines.append(
            "Use this causal structure as the basis of the answer. Never recommend "
            "an option marked INFEASIBLE merely because it is pleasant or convenient."
        )
        return "\n".join(lines)

    def unique_feasible_option(self) -> OptionAssessment | None:
        feasible = [option for option in self.options if option.feasible]
        return feasible[0] if len(feasible) == 1 else None


def analysis_prompt(text: str, agent_name: str = "") -> str:
    """Return a short, history-free prompt for executable goal analysis."""
    clean_agent_name = " ".join(str(agent_name or "").strip().split())
    named_agent_rule = (
        f'The human decision-maker is named "{clean_agent_name}". Use that exact name '
        "in semantic_roles.agent; never call them merely agent or user."
        if clean_agent_name else
        "Name the concrete decision-maker when the message provides their identity."
    )
    return f"""Analyze this practical decision using semantic roles and Boolean feasibility. Do not answer the user.

User message:
{text}

{named_agent_rule}

Extract the PRIMARY ACTION from the intended outcome, not from the means or
travel options. Extract its agent, patient/theme (the person or object acted
upon), required resources, and execution location. A patient/theme that is
transformed, cleaned, repaired, delivered, transported, installed, used, or
inspected is an indispensable participant.

For EACH option, simulate separately where the agent and patient/theme end up.
Describe the physical state transition caused by the option: what moves, what
stays behind, and which indispensable participant reaches the execution
location. Going somewhere transports the agent only unless the option itself
explicitly transports the patient/theme. For example, if a bicycle must be
repaired at a workshop, walking to the workshop does not bring the bicycle;
taking the bicycle does. Apply this relation to any domain, not just transport.
Also preserve temporal order: record where each participant is at the triggering
event and whether that participant can physically reach the required later
location in the available time. Never assume simultaneous incompatible locations,
an unmentioned helper, or an unmentioned remote-release mechanism.
Set these Boolean fields independently before setting feasible:
- patient_must_move_to_execution_location (once for the whole decision)
- moves_patient_or_theme (once for each option; derive patient location from it)
- agent_at_execution_location
- patient_at_execution_location
- required_resources_available
- agent_capability_sufficient

Agent capability includes reaction time, starting speed, acceleration, strength,
endurance, anatomy, traversable route, environmental limits, and safety. A merely
computable required performance is not feasible when the stated agent cannot
physically produce it. Feasible must be the logical AND of all four fields. Do not invent a later
transport, helper, tool, service, or action absent from the message. Weather,
comfort, speed, distance, and enjoyment are evaluated only after this test.
Keep every text field short and use the same language as the user.

Return only JSON with this exact shape:
{{
  "primary_goal": "short concrete outcome",
  "semantic_roles": {{
    "agent": "agent",
    "action": "primary action",
    "patient_or_theme": "object or person acted upon",
    "required_resources": ["resource"],
    "execution_location": "location"
  }},
  "patient_must_move_to_execution_location": true,
  "required_conditions": ["condition"],
  "options": [
    {{
      "name": "option as stated",
      "moves_patient_or_theme": true,
      "agent_at_execution_location": true,
      "patient_at_execution_location": true,
      "required_resources_available": true,
      "agent_capability_sufficient": true,
      "feasible": true,
      "reason": "short causal reason"
    }}
  ],
  "recommended_option": "one feasible option, or ask for clarification",
  "decisive_reason": "the condition that decides the choice"
}}"""


def parse_analysis(raw: str) -> GoalMeansAnalysis | None:
    """Parse and normalize a model-produced goal-means analysis."""
    cleaned = re.sub(r"^\s*```(?:json)?|```\s*$", "", str(raw or "").strip(), flags=re.I)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start:end + 1]
    try:
        data = json.loads(cleaned)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    goal = str(data.get("primary_goal") or "").strip()
    if not goal:
        return None
    conditions = [
        str(item).strip() for item in data.get("required_conditions", [])
        if str(item).strip()
    ][:8]
    semantic_roles = data.get("semantic_roles", {})
    if not isinstance(semantic_roles, dict):
        semantic_roles = {}
    semantic_roles = {
        str(key): str(value).strip()
        for key, value in semantic_roles.items()
        if str(value).strip()
    }
    options = []

    def clean_option_name(value: object) -> str:
        name = str(value or "").strip()
        # Keep the model's semantic option while normalizing a frequent French
        # finite-form leak into a phrase that can be quoted naturally.
        return re.sub(r"^(?:je\s+|j['’]\s*)?y\s+vais\b", "y aller", name, flags=re.I)

    def optional_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().casefold()
            if lowered in {"true", "yes", "1", "feasible"}:
                return True
            if lowered in {"false", "no", "0", "infeasible"}:
                return False
        return None

    patient_must_move = optional_bool(
        data.get("patient_must_move_to_execution_location")
    )

    for item in data.get("options", [])[:8]:
        if not isinstance(item, dict) or not str(item.get("name") or "").strip():
            continue
        agent_present = optional_bool(item.get("agent_at_execution_location"))
        patient_present = optional_bool(item.get("patient_at_execution_location"))
        moves_patient = optional_bool(item.get("moves_patient_or_theme"))
        resources_present = optional_bool(item.get("required_resources_available"))
        capability_sufficient = optional_bool(item.get("agent_capability_sufficient"))
        if patient_must_move is True:
            if moves_patient is False:
                patient_present = False
            elif moves_patient is True:
                patient_present = True
        prerequisite_checks = (agent_present, patient_present, resources_present)
        if capability_sufficient is not None:
            prerequisite_checks += (capability_sufficient,)
        if all(value is not None for value in prerequisite_checks):
            # The model identifies semantic roles; code owns the Boolean logic.
            feasible = all(prerequisite_checks)
        else:
            feasible = bool(optional_bool(item.get("feasible", False)))
        options.append(OptionAssessment(
            name=clean_option_name(item["name"]),
            feasible=feasible,
            reason=str(item.get("reason") or "No causal reason supplied").strip(),
            agent_at_execution_location=agent_present,
            patient_at_execution_location=patient_present,
            moves_patient_or_theme=moves_patient,
            required_resources_available=resources_present,
            agent_capability_sufficient=capability_sufficient,
        ))

    # Material-continuity repair. Analysis models can correctly name the acted-on
    # object yet still mark every travel option feasible. When one option moves
    # only the person and another explicitly includes that object, location is a
    # causal fact, not a preference for the response model to arbitrate.
    patient_text = semantic_roles.get("patient_or_theme", "")
    action_text = semantic_roles.get("action", "") or goal
    ignored_patient_words = {
        "a", "an", "the", "le", "la", "les", "un", "une", "du", "de",
        "des", "object", "objet", "thing", "chose", "patient", "theme",
    }
    patient_words = set(_normalized_words(patient_text).split()) - ignored_patient_words
    person_only_markers = (
        "a pied", "on foot", "walk", "walking", "seul", "seule", "alone",
    )
    person_only_options = []
    patient_carrying_options = []
    for option in options:
        option_words = set(_normalized_words(option.name).split())
        normalized_option = _normalized_words(option.name)
        if any(marker in normalized_option for marker in person_only_markers):
            person_only_options.append(option)
        if patient_words and patient_words.intersection(option_words):
            patient_carrying_options.append(option)

    if patient_words and person_only_options and patient_carrying_options:
        patient_must_move = True
        normalized_goal = _normalized_words(goal)
        french_goal = bool(re.search(
            r"\b(le|la|les|un|une|dois|devrais|pour|avec|chez)\b",
            normalized_goal,
        ))
        continuity_reason = (
            f"La présence de {patient_text} est indispensable pour {action_text}"
            if french_goal else
            f"The presence of {patient_text} is required for {action_text}"
        )
        for option in person_only_options:
            option.moves_patient_or_theme = False
            option.patient_at_execution_location = False
            option.feasible = False
            option.reason = continuity_reason
        for option in patient_carrying_options:
            option.moves_patient_or_theme = True
            option.patient_at_execution_location = True
            checks = (
                option.agent_at_execution_location,
                option.patient_at_execution_location,
                option.required_resources_available,
                option.agent_capability_sufficient,
            )
            known_checks = [value for value in checks if value is not None]
            if any(value is False for value in known_checks):
                option.feasible = False
            elif known_checks:
                option.feasible = True
            option.reason = continuity_reason
    recommendation = str(data.get("recommended_option") or "").strip()
    feasible_options = [option for option in options if option.feasible]
    recommended_match = next(
        (option for option in options if option.name.casefold() == recommendation.casefold()),
        None,
    )
    if len(feasible_options) == 1:
        recommendation = feasible_options[0].name
        decisive_reason = feasible_options[0].reason or str(
            data.get("decisive_reason") or ""
        ).strip()
    elif recommended_match is not None and not recommended_match.feasible:
        recommendation = feasible_options[0].name if feasible_options else "ask for clarification"
        decisive_reason = str(data.get("decisive_reason") or "").strip()
    else:
        decisive_reason = str(data.get("decisive_reason") or "").strip()
    return GoalMeansAnalysis(
        primary_goal=goal,
        semantic_roles=semantic_roles,
        patient_must_move_to_execution_location=patient_must_move,
        required_conditions=conditions,
        options=options,
        recommended_option=recommendation,
        decisive_reason=decisive_reason,
    )


def _normalized_words(value: str) -> str:
    folded = unicodedata.normalize("NFKD", str(value or "").casefold())
    ascii_text = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return " ".join(re.findall(r"[a-z0-9]+", ascii_text))


def _mentions_option(text: str, option_name: str) -> bool:
    haystack = f" {_normalized_words(text)} "
    option = _normalized_words(option_name)
    if not option:
        return False
    if f" {option} " in haystack:
        return True
    # Analysis models sometimes return a full infinitive phrase while the
    # answer uses its discriminating tail ("y aller a pied" -> "a pied").
    words = option.split()
    return len(words) >= 2 and f" {' '.join(words[-2:])} " in haystack


def _option_position(normalized_text: str, option_name: str) -> int:
    option = _normalized_words(option_name)
    markers = [option]
    words = option.split()
    if len(words) >= 2:
        markers.append(" ".join(words[-2:]))
    positions = [normalized_text.find(marker) for marker in markers if marker]
    positions = [position for position in positions if position >= 0]
    return min(positions) if positions else -1


def guarded_response(
    candidate: str,
    analysis: GoalMeansAnalysis,
    user_text: str,
    continuation: bool = False,
) -> tuple[str, bool]:
    """Prevent the response model from reversing a validated unique choice.

    The guard is intentionally narrow: it only acts when Boolean prerequisite
    checks leave exactly one feasible option. Ambiguous decisions remain the
    language model's responsibility.
    """
    selected = analysis.unique_feasible_option()
    if selected is None:
        return candidate, False

    visible = str(candidate or "").strip()
    lead = visible[:500]
    normalized_lead = _normalized_words(lead)
    raw_agent_name = str(analysis.semantic_roles.get("agent", "") or "").strip()
    generic_agents = {
        "", "agent", "user", "the user", "utilisateur", "l utilisateur",
        "human", "person", "personne", "je", "i",
    }
    agent_name = (
        raw_agent_name
        if _normalized_words(raw_agent_name) not in generic_agents else ""
    )
    names_agent = not agent_name or _normalized_words(agent_name) in normalized_lead
    recommendation_cue = re.search(
        r"\b(recommand|conseill|devr(?:ais|iez)|choisi|recommend|should|choose|best option)",
        normalized_lead,
    )
    recommended_is_present = _mentions_option(lead, selected.name)
    recommends_infeasible = False
    if recommendation_cue:
        recommendation_window = normalized_lead[recommendation_cue.end():]
        mentioned = [
            (_option_position(recommendation_window, option.name), option)
            for option in analysis.options
        ]
        mentioned = [(position, option) for position, option in mentioned if position >= 0]
        if mentioned:
            recommends_infeasible = not min(mentioned, key=lambda item: item[0])[1].feasible

    mentions_infeasible = any(
        _mentions_option(lead, option.name)
        for option in analysis.options
        if not option.feasible
    )
    reopens_settled_choice = "?" in visible or mentions_infeasible

    if (
        not continuation
        and visible
        and recommended_is_present
        and not recommends_infeasible
        and not reopens_settled_choice
        and names_agent
    ):
        return candidate, False

    reason = selected.reason or analysis.decisive_reason
    french = bool(re.search(
        r"\b(je|tu|vous|nous|dois|devrais|est ce|quelle|comment|avec|chez)\b",
        _normalized_words(user_text),
    ))
    if continuation and french:
        answer = "Exactement."
        if reason:
            answer += f" {reason.rstrip('.')} ."
    elif continuation:
        answer = "Exactly."
        if reason:
            answer += f" {reason.rstrip('.')} ."
    elif french:
        normalized_name = _normalized_words(selected.name)
        address = f"{agent_name}, " if agent_name else ""
        if normalized_name.startswith("y aller "):
            answer = f"{address}je recommande d'**{selected.name}**."
        elif normalized_name.startswith(("a pied", "en ", "avec ")):
            answer = f"{address}je recommande d'**y aller {selected.name}**."
        elif re.match(r"^[a-z]+(?:er|ir|re)\b", normalized_name):
            answer = f"{address}je recommande de **{selected.name}**."
        else:
            answer = f"{address}je recommande l'option **{selected.name}**."
        if not address:
            answer = answer[0].upper() + answer[1:]
        if reason:
            answer += f" {reason.rstrip('.')} ."
    else:
        address = f"{agent_name}, " if agent_name else ""
        answer = f"{address}I recommend **{selected.name}**."
        if reason:
            answer += f" {reason.rstrip('.')} ."
    return answer.replace(" .", "."), True


def is_analysis_followup(text: str, analysis: GoalMeansAnalysis) -> bool:
    """Recognize a nearby confirmation or correction about the same choice."""
    normalized = _normalized_words(text)
    if not normalized:
        return False
    if any(_mentions_option(normalized, option.name) for option in analysis.options):
        return True
    ignored = {
        "a", "an", "the", "to", "it", "i", "we", "you", "my", "your",
        "le", "la", "les", "un", "une", "de", "du", "des", "je", "tu",
        "vous", "nous", "mon", "ma", "mes", "et", "ou", "si", "pas",
    }
    context_words = set(_normalized_words(analysis.primary_goal).split())
    context_words.update(
        word
        for option in analysis.options
        for word in _normalized_words(option.name).split()
    )
    context_words -= ignored
    message_words = set(normalized.split()) - ignored
    return bool(context_words & message_words) and len(normalized.split()) <= 18
