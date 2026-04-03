import pytest

from tests.ut.utils import judge_expression


class TestVLAActionHeadSmoke:
    def test_action_head_compute_loss_smoke(self):
        torch = pytest.importorskip("torch")
        from mindspeed_mm.models.action.action_head import ActionHead
        cfg = {"hidden_layout": "sbh", "action_dim": 7, "action_horizon": 8, "state_dim": 8}
        head = ActionHead(cfg, text_hidden_size=32)
        hidden_states = torch.randn(12, 2, 32)
        actions = torch.randn(2, 8, 7)
        states = torch.randn(2, 1, 8)
        loss = head.compute_loss(hidden_states=hidden_states, actions=actions, state=states)
        judge_expression(loss.ndim == 0)
        judge_expression(bool(torch.isfinite(loss)))

    def test_action_head_compute_loss_shape_guard(self):
        torch = pytest.importorskip("torch")
        from mindspeed_mm.models.action.action_head import ActionHead
        cfg = {"hidden_layout": "sbh", "action_dim": 7, "action_horizon": 8}
        head = ActionHead(cfg, text_hidden_size=16)
        hidden_states = torch.randn(4, 2, 16)
        actions = torch.randn(2, 9, 7)
        with pytest.raises(ValueError):
            head.compute_loss(hidden_states=hidden_states, actions=actions)
