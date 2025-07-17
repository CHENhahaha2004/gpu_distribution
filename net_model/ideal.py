import simpy
from .base import BaseNetModel


class IdealNet(BaseNetModel):
    """Ideal network model: no contention; time = size / bandwidth."""

    def transfer(self, src: int, dst: int, size: int, bw: float):  # noqa: D401
        duration = size / bw if bw > 0 else 0.0
        yield self.env.timeout(duration) 