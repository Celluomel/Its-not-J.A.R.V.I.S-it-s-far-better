"""
network_mapping.py — 7 Functional Networks (Yeo 2011 canonical)
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ── 7 Yeo networks ────────────────────────────────────────────────────────────
NETWORKS = [
    "DMN",   # Default Mode Network
    "DAN",   # Dorsal Attention
    "VAN",   # Ventral Attention
    "SMN",   # Somatomotor
    "VIS",   # Visual
    "LIM",   # Limbic
    "FPN",   # Frontoparietal
]

NEON_COLORS: Dict[str, Tuple[float, float, float]] = {
    "DMN": (0.80, 0.40, 1.00),   # neon violet    — Inner Monologue
    "DAN": (0.20, 0.80, 1.00),   # neon cyan-blue — Curiosity
    "VAN": (1.00, 0.60, 0.10),   # neon orange    — Surprise/Salience
    "SMN": (0.20, 1.00, 0.90),   # neon mint      — Expression/Output
    "VIS": (1.00, 0.90, 0.20),   # neon yellow    — Perception
    "LIM": (1.00, 0.30, 0.80),   # neon magenta   — Social/Relational
    "FPN": (0.30, 1.00, 0.50),   # neon green     — Goals/Executive
}

# 180 synthetic regions distributed across 7 networks
_NET_REGIONS = {
    "DMN": range(0,   26),
    "DAN": range(26,  52),
    "VAN": range(52,  78),
    "SMN": range(78,  104),
    "VIS": range(104, 130),
    "LIM": range(130, 156),
    "FPN": range(156, 180),
}

def build_region_to_network_map(max_region_id: int) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for rid in range(max_region_id + 1):
        net = NETWORKS[rid % len(NETWORKS)]  # fallback for out-of-range
        for n, rng in _NET_REGIONS.items():
            if rid in rng: net = n; break
        mapping[rid] = net
    return mapping

@dataclass
class BrainState:
    """Holds activation [0–1] for each of the 7 functional networks."""
    activations: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        for net in NETWORKS:
            if net not in self.activations:
                self.activations[net] = 0.0

    def set_region(self, network_name: str, value: float):
        if network_name not in NETWORKS:
            raise ValueError(f"Unknown network: {network_name}. Valid: {NETWORKS}")
        self.activations[network_name] = max(0.0, min(1.0, float(value)))

    def get_region_activation(self, network_name: str) -> float:
        return self.activations.get(network_name, 0.0)

    def as_uniform_array(self) -> List[float]:
        return [self.activations[net] for net in NETWORKS]

def create_default_brain_state() -> BrainState:
    return BrainState()

def create_region_network_mapping(region_ids: List[int]) -> Dict[int, str]:
    if not region_ids: return {}
    return build_region_to_network_map(max(region_ids))
