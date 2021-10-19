
import random
from .spos_mobilenet_v3 import SPOSMobileNetV3

class LegencyOFAArchitecture:
    def __init__(self, ks, ratios, depths) -> None:
        self.ks = ks
        self.ratios = ratios
        self.depths = depths


class LegencyOFAArchitectureGenerator:
    def __init__(self, ks_candidates=[3, 5, 7], ratios_candidates=[3, 4, 6], depths_candidates=[2, 3, 4]) -> None:
        self.ks_candidates = ks_candidates
        self.ratios_candidates = ratios_candidates
        self.depths_candidates = depths_candidates

    def random(self):
        ks = [random.choice(self.ks_candidates) for _ in range(20)]
        ratios = [random.choice(self.ratios_candidates) for _ in range(20)]
        depths = [random.choice(self.depths_candidates) for _ in range(5)]
        return LegencyOFAArchitecture(ks, ratios, depths)
