"""Grounded quantitative constraints for conversational problem solving."""
from __future__ import annotations

import math
import re
import unicodedata
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from dataclasses import dataclass, field
from typing import List, Optional


def _normalized(text: str) -> str:
    folded = unicodedata.normalize("NFKD", str(text or "").casefold())
    return " ".join(
        "".join(ch for ch in folded if not unicodedata.combining(ch)).split()
    )


def _contains_approximate_token(text: str, targets: tuple[str, ...]) -> bool:
    """Match common transcription and typing errors conservatively."""
    tokens = re.findall(r"[a-z]+", text)
    return any(
        SequenceMatcher(None, token, target).ratio() >= 0.72
        for token in tokens
        for target in targets
    )


@dataclass
class QuantitativeAnalysis:
    target: str
    knowns: List[str] = field(default_factory=list)
    unknowns: List[str] = field(default_factory=list)
    relation: str = ""
    conclusion: str = ""
    concise_answer: str = ""
    fallback_answer: str = ""
    agent: str = "unspecified"
    capability_constraints: List[str] = field(default_factory=list)
    feasible: Optional[bool] = None

    def answer_for_verbosity(self, verbosity: str = "concise") -> str:
        """Return a conversational verdict or the full derivation."""
        if str(verbosity or "").casefold() == "verbose":
            return self.fallback_answer
        return self.concise_answer or self.fallback_answer

    def prompt_fragment(self) -> str:
        return "\n".join([
            "━━ QUANTITATIVE CONSTRAINT ANALYSIS ━━",
            f"Target quantity: {self.target}",
            "Known facts: " + ("; ".join(self.knowns) or "none"),
            "Missing inputs: " + ("; ".join(self.unknowns) or "none"),
            f"Governing relation: {self.relation}",
            f"Agent: {self.agent}",
            "Agent capability constraints: "
            + ("; ".join(self.capability_constraints) or "none established"),
            "Physical feasibility: "
            + ("INFEASIBLE" if self.feasible is False else "UNDETERMINED"),
            f"Grounded conclusion: {self.conclusion}",
            "Answer with the calculation or symbolic relation first. Distinguish explicit facts "
            "from assumptions. If inputs are missing, do not invent them and do not replace the "
            "unknown result with a tautology. A mathematically definable speed is not a valid "
            "recommendation when it exceeds the actor's physical capability or requires an "
            "unsafe path. State physical infeasibility first. Never expose semantic-role labels "
            "such as Agent, Patient, or Theme.",
        ])


@dataclass
class TemporalTravelAnalysis:
    """Grounded deadline reasoning for travel questions.

    A phrase such as "there in an hour" describes an arrival deadline, not
    an exact departure time.  Departure can only be calculated after a route
    duration is known, so this object keeps those two quantities separate.
    """

    now: datetime
    arrival_deadline: datetime
    duration_text: str
    concise_answer: str
    detailed_answer: str

    def answer_for_verbosity(self, verbosity: str = "concise") -> str:
        return self.detailed_answer if str(verbosity or "").casefold() == "verbose" else self.concise_answer

    def prompt_fragment(self) -> str:
        return "\n".join([
            "━━ TEMPORAL DEADLINE ANALYSIS ━━",
            f"Local reference time: {self.now.strftime('%H:%M')}",
            f"Arrival deadline: {self.arrival_deadline.strftime('%H:%M')}",
            f"Stated duration: {self.duration_text}",
            "The duration refers to the latest arrival time, not an exact departure time.",
            "Do not invent a route duration, traffic estimate, or departure time. "
            "If travel duration is T, departure deadline is arrival deadline minus T. "
            "Keep 12-hour and 24-hour notation unambiguous.",
        ])


def analyze_temporal_travel_question(text: str) -> Optional[TemporalTravelAnalysis]:
    """Resolve relative arrival deadlines without asking the generative LLM."""
    normalized = _normalized(text)
    travel = bool(re.search(
        r"\b(?:aller|rendre|arriver|etre|y etre|go|reach|get)\b", normalized
    )) and bool(re.search(
        r"\b(?:saint denis|destination|dans\s+\d+\s*(?:heure|heures|minute|minutes)|in\s+\d+\s*(?:hour|hours|minute|minutes))\b",
        normalized,
    ))
    duration_match = re.search(
        r"\bdans\s+(une|un|\d+)\s*(heure|heures|h|minute|minutes)\b|"
        r"\bin\s+(an|one|\d+)\s*(hour|hours|minute|minutes)\b",
        normalized,
    )
    if not (travel and duration_match):
        return None

    amount_word = duration_match.group(1) or duration_match.group(3)
    unit = duration_match.group(2) or duration_match.group(4)
    amount = 1 if amount_word in {"une", "un", "an", "one"} else int(amount_word)
    minutes = amount * (60 if unit.startswith(("heure", "hour", "h")) else 1)
    now = datetime.now().astimezone()
    deadline = now + timedelta(minutes=minutes)
    now_label = now.strftime("%Hh%M")
    deadline_label = deadline.strftime("%Hh%M")
    duration_label = f"{minutes} minute" if minutes == 1 else f"{minutes} minutes"
    concise = (
        f"L'échéance d'arrivée est {deadline_label} ({duration_label} à partir de maintenant). "
        "L'heure de départ dépend de la durée réelle du trajet : partez à "
        "l'heure d'arrivée moins cette durée. Sans durée de trajet fiable, on ne peut pas "
        "donner une heure de départ exacte."
    )
    detailed = (
        f"Il est actuellement {now_label}. « Être là dans {duration_label} » signifie arriver "
        f"au plus tard vers {deadline_label}, et non partir à {deadline_label}. "
        "Pour calculer le départ, il faut la durée du trajet : départ = heure d'arrivée "
        "moins durée du trajet. Par exemple, pour un trajet de 45 minutes, partez 45 minutes "
        f"avant {deadline_label}; pour 60 minutes, partez à {now_label}. Je ne dois pas "
        "inventer une durée ou confondre 01h03 avec 12h03. Vérifiez la durée actuelle sur "
        "votre outil de trajet et ajoutez une marge si nécessaire."
    )
    return TemporalTravelAnalysis(now, deadline, duration_label, concise, detailed)


def analyze_quantitative_question(
    text: str,
    agent_name: str = "",
) -> Optional[QuantitativeAnalysis]:
    """Recognize supported quantitative constraints without another LLM call."""
    normalized = _normalized(text)
    asks_speed = bool(re.search(
        r"\b(quelle vitesse|a quelle vitesse|vitesse dois je|how fast|what speed)\b",
        normalized,
    ))
    falling_intercept = (
        any(word in normalized for word in ("lache", "tombe", "chute", "falling", "drop"))
        and _contains_approximate_token(
            normalized, ("rattraper", "attraper", "catch", "intercept")
        )
    )
    if not (asks_speed and falling_intercept):
        return None

    floor_match = re.search(
        r"\b(?:du|de\s+la|from\s+the)?\s*(\d+)\s*(?:e|eme|ieme|nd|rd|th)?\s*(?:etage|floor)\b",
        normalized,
    )
    floor = int(floor_match.group(1)) if floor_match else None
    assumed_height = float(floor * 3) if floor and floor > 0 else None
    fall_time = math.sqrt(2.0 * assumed_height / 9.81) if assumed_height else None
    same_person_releases = bool(re.search(
        r"\b(?:que\s+)?je\s+lache\b|\bi\s+(?:drop|release)\b",
        normalized,
    ))
    knowns = ["the object is released and falls under gravity"]
    if floor:
        knowns.append(f"the release point is described as floor {floor}")
    if same_person_releases:
        knowns.append(
            "the runner releases the object and is at the upper release point at t=0"
        )
    unknowns = [
        "the actual vertical drop height h",
        "the traversable route distance d from the release point to the interception point",
        "reaction time t_r",
    ]
    relation = "t_fall = sqrt(2h/g); required average speed v >= d / (t_fall - t_r)"
    capability_constraints = [
        "a human starts from rest and has a non-zero reaction time",
        "running between floors requires a longer constrained route with turns or stairs",
        "human locomotion cannot safely match free-fall acceleration on that route",
    ]
    if fall_time is not None:
        approximation = (
            f"Using only an illustrative 3 m per floor assumption gives h ~= "
            f"{assumed_height:.0f} m and t_fall ~= {fall_time:.1f} s."
        )
    else:
        approximation = "No numerical fall time can be computed until h is known."
    if same_person_releases:
        conclusion = (
            "No physically achievable human running speed satisfies the request under the stated setup. "
            "The route is not a horizontal sprint: the same person starts at the upper release "
            "point at t=0 and must reach the ground through a longer constrained path after release, "
            "while the object takes the direct free-fall path. "
            + approximation
        )
    else:
        conclusion = (
            "There is no unique numerical speed because route distance, initial runner location, "
            "and reaction time are missing. " + approximation
        )
    clean_agent_name = " ".join(str(agent_name or "").strip().split())
    agent_label = clean_agent_name or "the human runner"
    french = bool(re.search(r"\b(je|objet|etage|vitesse|courir|immeuble)\b", normalized))
    if french:
        address = f"{clean_agent_name}, " if clean_agent_name else ""
        impossible_opening = (
            "aucune vitesse de course humainement réalisable ne permet de rattraper l'objet "
            if address else
            "Aucune vitesse de course humainement réalisable ne permet de rattraper l'objet "
        )
        unknown_opening = (
            "impossible de déterminer une vitesse réalisable avec ces seules données. "
            if address else
            "Impossible de déterminer une vitesse réalisable avec ces seules données. "
        )
        floor_label = f"au {floor}e étage" if floor else "au point de lâcher en hauteur"
        if fall_time is not None:
            french_approximation = (
                f"En supposant seulement 3 m par étage pour obtenir un ordre de grandeur, "
                f"la hauteur serait d'environ {assumed_height:.0f} m et la chute durerait "
                f"environ {fall_time:.1f} seconde."
            )
        else:
            french_approximation = (
                "Il faut connaître la hauteur h pour calculer précisément le temps de chute."
            )
        if same_person_releases:
            concise = (
                address
                + (
                    "ce n'est pas physiquement réalisable. "
                    if address else
                    "Ce n'est pas physiquement réalisable. "
                )
                + f"Comme c'est toi qui lâches l'objet, tu te trouves encore {floor_label} "
                "quand il commence à tomber. "
                + (
                    f"Il atteindra le sol en environ {fall_time:.1f} seconde, trop vite pour "
                    if fall_time is not None else
                    "Il tombera trop vite pour "
                )
                + "ton temps de réaction et la descente du bâtiment. Aucune vitesse de course "
                  "humaine ne suffit; ne tente pas cette expérience."
            )
        else:
            concise = (
                address
                + (
                    "impossible de calculer une vitesse réaliste avec ces seules données. "
                    if address else
                    "Impossible de calculer une vitesse réaliste avec ces seules données. "
                )
                + "Il manque la hauteur, la position initiale de la personne, son temps de "
                  "réaction et la longueur du trajet praticable."
            )
        fallback = (
            address
            + (
                impossible_opening + "dans cette configuration. "
                if same_person_releases else
                unknown_opening
            )
            + french_approximation + " "
            "La relation est t = sqrt(2h/g), puis v >= d/(t - t_r), où d est la longueur "
            "du trajet praticable entre le point de lâcher et le point d'interception, pas une "
            "distance horizontale inventée, et t_r ton temps de réaction. "
            + (
                f"Puisque c'est toi qui lâches l'objet, tu te trouves encore {floor_label} à t=0 : "
                "tu pars à l'arrêt, avec un temps de réaction, puis dois descendre un trajet plus "
                "long et contraint pendant que l'objet suit directement la chute libre. Tu devrais "
                f"effectuer ce trajet en moins d'environ {fall_time:.1f} seconde, ce qui dépasse les capacités "
                "physiques humaines dans un immeuble. "
                if same_person_releases and fall_time is not None else
                "La position initiale et le trajet doivent être précisés avant tout résultat numérique. "
            )
            + (
                "Le rattrapage est irréaliste et dangereux : ne tente pas l'expérience."
                if same_person_releases else
                "Il faut notamment préciser qui lâche l'objet et où se trouve la personne qui le rattrape."
            )
        )
    else:
        english_address = f"{clean_agent_name}, " if clean_agent_name else ""
        if same_person_releases:
            concise = (
                english_address
                + (
                    "this is not physically achievable. "
                    if english_address else
                    "This is not physically achievable. "
                )
                + "Because you release the object, you are still upstairs when it starts "
                  "falling. It reaches the ground too quickly for human reaction time and a "
                  "descent through the building. No realistic human running speed is enough; "
                  "do not attempt it."
            )
        else:
            concise = (
                english_address
                + (
                    "there is not enough information to calculate an achievable speed. "
                    if english_address else
                    "There is not enough information to calculate an achievable speed. "
                )
                + "The height, runner's initial position, reaction time, and traversable route "
                  "length are missing."
            )
        fallback = (
            english_address
            + (
                "No physically achievable human running speed can catch the object in this setup. "
                if same_person_releases else
                "There is not enough information to determine an achievable running speed. "
            )
            + approximation + " "
            + "The relation is t = sqrt(2h/g), then v >= d/(t - t_r), where d is the "
            "traversable route from release point to interception point, not an invented "
            "horizontal distance. "
            + (
                "Since the same person releases the object upstairs, they start upstairs at t=0 "
                "and would have to descend that route within the fall time. No realistic running "
                "speed achieves that in a building; do not attempt it."
                if same_person_releases else
                "The release mechanism and the runner's initial location must be specified."
            )
        )
    return QuantitativeAnalysis(
        target="minimum average running speed before the object reaches the ground",
        knowns=knowns,
        unknowns=unknowns,
        relation=relation,
        conclusion=conclusion,
        concise_answer=concise,
        fallback_answer=fallback,
        agent=agent_label,
        capability_constraints=capability_constraints,
        feasible=False if same_person_releases else None,
    )


def guard_quantitative_response(
    candidate: str,
    analysis: QuantitativeAnalysis,
    verbosity: str = "verbose",
) -> tuple[str, bool]:
    """Replace circular or leaked-internal answers with the computed constraint."""
    visible = str(candidate or "").strip()
    normalized = _normalized(visible)
    leaks_roles = bool(re.search(r"\b(agent|patient|theme)\b", normalized))
    tautology = any(phrase in normalized for phrase in (
        "vitesse necessaire", "required speed", "speed needed", "fast enough",
    ))
    shows_reasoning = (
        any(marker in visible for marker in ("sqrt", "√", ">="))
        and any(word in normalized for word in ("distance", "reaction", "donnee", "information"))
    )
    invents_horizontal_route = bool(re.search(
        r"\b(distance horizontale|horizontal distance|horizontal sprint)\b", normalized
    ))
    models_world_state = (
        any(phrase in normalized for phrase in (
            "meme personne", "c est toi qui", "tu es encore", "same person", "start upstairs",
        ))
        and any(word in normalized for word in (
            "trajet", "descendre", "escalier", "route", "descend",
        ))
    )
    states_human_limit = any(phrase in normalized for phrase in (
        "aucune vitesse", "impossible physiquement", "capacites physiques humaines",
        "no physically achievable", "humanly impossible", "beyond human",
    ))
    if (
        visible and shows_reasoning and models_world_state and states_human_limit
        and not leaks_roles and not tautology and not invents_horizontal_route
    ):
        return candidate, False
    return analysis.answer_for_verbosity(verbosity), True
