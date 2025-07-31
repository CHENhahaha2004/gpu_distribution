import simpy
from .base import BaseNetModel
#理想网络模型：无竞争；时间 = 大小 / 带宽。

class IdealNet(BaseNetModel):
    """Ideal network model: no contention; time = size / bandwidth."""

    def transfer(self, src: int, dst: int, size: int, bw: float, *, priority: int = 2):  # noqa: D401
        duration = size / bw if bw > 0 else 0.0
        yield self.env.timeout(duration)
        # 返回空字典，保持与 RDMA 接口一致
        return {} 