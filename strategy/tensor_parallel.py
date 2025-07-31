import simpy
from typing import List

from .base import ParallelStrategy
from ..simulator.gpu import GPU


class TensorParallelStrategy(ParallelStrategy):
    """Tensor parallelism: split tensor, partial compute, then all-reduce."""

    def run(self, env: simpy.Environment, gpus: List[GPU], recorder):
        from collections import defaultdict

        NUM_LAYERS = 4  # 实验固定 4 层
        num_parts = len(gpus)
        flop_per_layer_total = self.flops_per_batch / NUM_LAYERS  # 整层 FLOPs
        partial_flop = flop_per_layer_total / num_parts          # 每 GPU 分片 FLOPs

        # 预计算节点分组和 leader（与 data-parallel 相同）
        node_to_gpus = defaultdict(list)
        for g in gpus:
            node_to_gpus[g.node_id].append(g)

        leaders = [grp[0] for grp in node_to_gpus.values()]

        for mb in range(self.micro_batches):
            # ------------------- 前向 -------------------
            for layer in range(NUM_LAYERS):
                # 1) 部分前向计算
                compute_events = [env.process(g.compute(f"mb{mb}_L{layer}_fwd_partial", partial_flop)) for g in gpus]
                yield simpy.events.AllOf(env, compute_events)

                # 2) All-Reduce 汇总前向输出（激活，priority=2）
                yield simpy.events.AllOf(env, self._hierarchical_all_reduce(env, gpus, node_to_gpus, leaders,
                                                                           data_type="activation", priority=2))

            # ------------------- 反向 -------------------
            for layer in reversed(range(NUM_LAYERS)):
                # 3) 部分反向计算
                compute_events = [env.process(g.compute(f"mb{mb}_L{layer}_bwd_partial", partial_flop)) for g in gpus]
                yield simpy.events.AllOf(env, compute_events)

                # 4) All-Reduce 汇总梯度（gradient，priority=1）
                yield simpy.events.AllOf(env, self._hierarchical_all_reduce(env, gpus, node_to_gpus, leaders,
                                                                           data_type="gradient", priority=1))

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------
    def _hierarchical_all_reduce(self, env: simpy.Environment, gpus: List[GPU], node_to_gpus, leaders,
                                 *, data_type: str, priority: int):
        """分层 All-Reduce：返回 simpy 事件列表。

        参数
        ------
        data_type : str
            "activation" 或 "gradient"。
        priority : int
            1=高 (梯度) / 2=中 (激活)。
        """
        events = []

        # 1) 节点内 reduce（非 leader → leader）
        for grp in node_to_gpus.values():
            leader = grp[0]
            for g in grp[1:]:
                events.append(env.process(g.send(leader.id, self.comm_size, data_type=data_type, priority=priority)))

        # 2) 节点间 ring（leaders 之间）
        if len(leaders) > 1:
            num_leaders = len(leaders)
            for idx, g in enumerate(leaders):
                dst = leaders[(idx + 1) % num_leaders].id
                events.append(env.process(g.send(dst, self.comm_size, data_type=data_type, priority=priority)))

        # 3) 节点内 broadcast（leader → 其他 GPU）
        for grp in node_to_gpus.values():
            leader = grp[0]
            for g in grp[1:]:
                events.append(env.process(leader.send(g.id, self.comm_size, data_type=data_type, priority=priority)))

        return events 