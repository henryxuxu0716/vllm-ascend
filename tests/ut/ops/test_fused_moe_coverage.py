#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""
Additional unit tests for the fused_moe module to cover paths
that were not exercised by the existing test suite after the
recent refactor commits (ae068a33, 63c363d3, 94dd8328, 4f937f56).
"""

import unittest
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch_npu
from pytest_mock import MockerFixture
from vllm.model_executor.layers.fused_moe import FusedMoEConfig

from tests.ut.base import TestBase
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe.experts_selector import (
    _native_grouped_topk, _native_select_experts, _renormalize_topk_weights,
    _select_expert_use_group_topk, _select_experts_with_fusion_ops,
    select_experts)
from vllm_ascend.ops.fused_moe.fused_moe import (
    AscendFusedMoE, AscendSharedFusedMoE, AscendUnquantizedFusedMoEMethod)
from vllm_ascend.ops.fused_moe.moe_comm_method import (AllGatherCommImpl,
                                                       get_moe_comm_method,
                                                       setup_moe_comm_method)
from vllm_ascend.ops.fused_moe.moe_mlp import (cumsum_group_list,
                                               unified_apply_mlp)
from vllm_ascend.ops.fused_moe.prepare_finalize import (
    PrepareAndFinalizeWithAll2All, PrepareAndFinalizeWithAllGather,
    PrepareAndFinalizeWithMC2, QuantType)
from vllm_ascend.utils import adapt_patch

adapt_patch(True)


# ──────────────────────────────────────────────────────────────────
# helpers / fixtures
# ──────────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def setup_vllm_config_mock(mocker: MockerFixture):
    mock_hf_config = MagicMock()
    mock_hf_config.model_type = "llama"

    mock_model_config = MagicMock()
    mock_model_config.hf_config = mock_hf_config

    mock_vllm_config = MagicMock()
    mock_vllm_config.model_config = mock_model_config
    mock_vllm_config.parallel_config = MagicMock(tensor_parallel_size=2)
    mock_vllm_config.scheduler_config = MagicMock(max_num_seqs=4)
    mock_vllm_config.model_config.max_model_len = 2048

    mocker.patch('vllm_ascend.ops.fused_moe.fused_moe.get_current_vllm_config',
                 return_value=mock_vllm_config)
    mocker.patch(
        'vllm_ascend.ops.fused_moe.moe_comm_method.get_current_vllm_config',
        return_value=mock_vllm_config)


def _mock_ep_group(mocker):
    g = mocker.MagicMock()
    g.rank_in_group = 0
    g.rank = 0
    g.world_size = 4
    g.device_group = "mock_ep"
    return g


def _mock_tp_group(mocker):
    g = mocker.MagicMock()
    g.rank_in_group = 0
    g.world_size = 2
    g.device_group = "mock_tp"
    return g


@pytest.fixture
def mock_dist_env(mocker: MockerFixture):
    """Set up a comprehensive distributed-environment mock."""
    mock_moe_comm = MagicMock()

    def _prepare(hidden_states, router_logits, **kw):
        return hidden_states, router_logits

    mock_moe_comm.prepare.side_effect = _prepare
    mock_moe_comm.fused_experts.return_value = torch.randn(16, 2)

    def _finalize(hidden_states, **kw):
        return hidden_states

    mock_moe_comm.finalize.side_effect = _finalize

    dp_metadata = MagicMock(num_tokens_across_dp_cpu=[5, 5])
    fwd_ctx = MagicMock(moe_comm_method=mock_moe_comm,
                        moe_comm_type=MoECommType.MC2,
                        max_tokens_across_dp=10,
                        dp_metadata=dp_metadata,
                        mc2_mask=torch.zeros(16, dtype=torch.bool),
                        padded_num_tokens=16,
                        with_quant=False,
                        weight_prefetch_method=MagicMock())

    with patch('torch.distributed.get_rank', return_value=0), \
        patch('torch.distributed.get_world_size', return_value=4), \
        patch('vllm_ascend.ops.fused_moe.fused_moe.get_ep_group',
              return_value=_mock_ep_group(mocker)), \
        patch('vllm_ascend.ops.fused_moe.token_dispatcher.get_ep_group',
              return_value=_mock_ep_group(mocker)), \
        patch('vllm_ascend.ops.fused_moe.fused_moe.get_mc2_group',
              return_value=_mock_ep_group(mocker)), \
        patch('vllm_ascend.ops.fused_moe.fused_moe.get_tp_group',
              return_value=_mock_tp_group(mocker)), \
        patch('vllm.distributed.parallel_state.get_tp_group',
              return_value=_mock_tp_group(mocker)), \
        patch('vllm_ascend.ops.fused_moe.fused_moe.get_dp_group',
              return_value=_mock_tp_group(mocker)), \
        patch('vllm.model_executor.layers.fused_moe.layer.get_dp_group',
              return_value=_mock_tp_group(mocker)), \
        patch('vllm.model_executor.layers.fused_moe.config.get_dp_group',
              return_value=_mock_tp_group(mocker)), \
        patch('vllm_ascend.ops.fused_moe.fused_moe.get_ascend_config',
              return_value=MagicMock(
                  torchair_graph_config=MagicMock(enabled=False),
                  enable_multistream_moe=False,
                  expert_map_path=None,
                  enable_shared_expert_dp=False,
                  dynamic_eplb=False,
                  init_redundancy_expert=0,
                  multistream_overlap_shared_expert=False,
              )), \
        patch('vllm_ascend.ops.fused_moe.fused_moe.determine_expert_map',
              return_value=(3, torch.tensor(
                  [0, 1, 2, -1, -1, -1, -1, -1]))), \
        patch('vllm_ascend.ops.fused_moe.fused_moe.get_forward_context',
              return_value=fwd_ctx), \
        patch('vllm_ascend.ops.fused_moe.prepare_finalize.get_forward_context',
              return_value=fwd_ctx), \
        patch("vllm_ascend.utils.get_ascend_soc_version",
              return_value=1), \
        patch('vllm_ascend.ops.fused_moe.moe_mlp.get_forward_context',
              return_value=fwd_ctx), \
        patch('vllm_ascend.ops.fused_moe.moe_comm_method.MC2CommImpl._get_token_dispatcher',
              return_value=None), \
        patch('vllm_ascend.ops.fused_moe.moe_comm_method.AlltoAllCommImpl._get_token_dispatcher',
              return_value=None), \
        patch('vllm_ascend.ops.fused_moe.moe_comm_method.AllGatherCommImpl._get_token_dispatcher',
              return_value=None), \
        patch('vllm_ascend.ops.fused_moe.experts_selector.get_forward_context',
              return_value=fwd_ctx):
        yield {
            'fwd_ctx': fwd_ctx,
            'mock_moe_comm': mock_moe_comm,
        }


@pytest.fixture
def mock_moe_env(mocker: MockerFixture):
    with patch('torch_npu.npu_moe_gating_top_k', return_value=(
            torch.randn(8, 2),
            torch.randint(0, 8, (8, 2)),
            None)), \
        patch('torch_npu.npu_moe_init_routing', return_value=(
            torch.randn(8, 2),
            torch.randint(0, 8, (8, 2)),
            torch.tensor([0, 1, 2, 4, 6, 2, 7, 1]))), \
        patch("torch_npu.npu_moe_compute_expert_tokens",
              return_value=torch.randn(8, 2)), \
        patch("torch_npu.npu_moe_distribute_dispatch",
              return_value=torch.randn(16, 2)), \
        patch("torch_npu.npu_moe_distribute_combine",
              return_value=torch.randn(16, 2)), \
        patch("torch_npu.npu_grouped_matmul",
              return_value=[torch.randn(16, 2)]), \
        patch("torch_npu.npu_swiglu",
              return_value=torch.randn(16, 2)), \
        patch("torch_npu.npu_moe_gating_top_k_softmax", return_value=(
            torch.randn(8, 2),
            torch.randint(0, 8, (8, 2)),
            torch.tensor([0, 1, 2, 4, 6, 2, 7, 1]))), \
        patch("torch_npu.npu_moe_finalize_routing",
              return_value=torch.randn(16, 2)):
        if hasattr(torch_npu, 'npu_moe_distribute_dispatch_v2'):
            with patch("torch_npu.npu_moe_distribute_dispatch_v2",
                       return_value=torch.randn(16, 2)), \
                 patch("torch_npu.npu_moe_distribute_combine_v2",
                       return_value=torch.randn(16, 2)):
                yield
        else:
            yield


# ──────────────────────────────────────────────────────────────────
# 1. experts_selector coverage
# ──────────────────────────────────────────────────────────────────
class TestNativeSelectExperts:
    """Cover _native_select_experts branches not hit by existing tests."""

    def test_softmax_scoring(self, mock_dist_env, mock_moe_env):
        """select_experts → _native_select_experts with softmax + custom_routing_function."""

        def custom_route(hidden_states, gating_output, topk, renormalize,
                         global_num_experts):
            w = torch.softmax(gating_output, dim=-1)
            topk_w, topk_i = w.topk(topk, dim=-1)
            return topk_w, topk_i

        x = torch.randn(4, 8)
        logits = torch.randn(4, 8)
        tw, ti = select_experts(hidden_states=x,
                                router_logits=logits,
                                top_k=2,
                                use_grouped_topk=False,
                                renormalize=True,
                                custom_routing_function=custom_route,
                                scoring_func="softmax",
                                global_num_experts=8)
        assert tw.shape == (4, 2)
        assert ti.shape == (4, 2)
        assert ti.dtype == torch.int32

    def test_sigmoid_scoring(self, mock_dist_env, mock_moe_env):
        """_native_select_experts with scoring_func='sigmoid'."""
        x = torch.randn(4, 8)
        logits = torch.randn(4, 8)
        tw, ti = _native_select_experts(hidden_states=x,
                                        router_logits=logits,
                                        top_k=2,
                                        use_grouped_topk=False,
                                        renormalize=True,
                                        scoring_func="sigmoid")
        assert tw.shape == (4, 2)
        assert ti.dtype == torch.int32

    def test_unsupported_scoring_raises(self, mock_dist_env, mock_moe_env):
        """_native_select_experts with invalid scoring_func raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported scoring function"):
            _native_select_experts(hidden_states=torch.randn(4, 8),
                                   router_logits=torch.randn(4, 8),
                                   top_k=2,
                                   use_grouped_topk=False,
                                   renormalize=True,
                                   scoring_func="invalid_func")

    def test_grouped_topk_path(self, mock_dist_env, mock_moe_env):
        """_native_select_experts with use_grouped_topk=True."""
        x = torch.randn(4, 16)
        logits = torch.randn(4, 16)
        tw, ti = _native_select_experts(hidden_states=x,
                                        router_logits=logits,
                                        top_k=2,
                                        use_grouped_topk=True,
                                        renormalize=True,
                                        topk_group=2,
                                        num_expert_group=4,
                                        scoring_func="softmax")
        assert tw.shape == (4, 2)
        assert ti.dtype == torch.int32


class TestNativeGroupedTopk:
    """Cover _native_grouped_topk directly."""

    def test_basic_grouped_topk(self):
        weights = torch.randn(4, 16)
        result = _native_grouped_topk(weights,
                                      num_expert_group=4,
                                      topk_group=2)
        assert result.shape == weights.shape
        # Masked entries should be zero
        assert (result == 0.0).any()

    def test_none_params_default_to_zero(self):
        weights = torch.randn(4, 8)
        # When None is passed, should default to 0 (handles edge case)
        result = _native_grouped_topk(weights,
                                      num_expert_group=None,
                                      topk_group=None)
        assert result.shape == weights.shape


class TestRenormalizeTopkWeights:
    """Cover _renormalize_topk_weights."""

    def test_renormalize_true(self):
        weights = torch.tensor([[0.3, 0.7], [0.5, 0.5]])
        result = _renormalize_topk_weights(weights, renormalize=True)
        # Each row should sum to 1
        row_sums = result.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(2), atol=1e-5)

    def test_renormalize_false(self):
        weights = torch.tensor([[0.3, 0.7], [0.2, 0.4]])
        result = _renormalize_topk_weights(weights, renormalize=False)
        assert torch.equal(result, weights)


class TestSelectExpertUseGroupTopk:
    """Cover _select_expert_use_group_topk branches."""

    def test_without_correction_bias(self):
        weights = torch.randn(4, 16).softmax(dim=-1)
        tw, ti = _select_expert_use_group_topk(topk_weights=weights,
                                               topk_group=2,
                                               renormalize=True,
                                               top_k=2,
                                               num_expert_group=4,
                                               e_score_correction_bias=None)
        assert tw.shape == (4, 2)
        assert ti.shape == (4, 2)
        assert ti.dtype == torch.int32

    def test_with_correction_bias(self):
        weights = torch.randn(4, 16).softmax(dim=-1)
        bias = torch.randn(16) * 0.1
        tw, ti = _select_expert_use_group_topk(topk_weights=weights,
                                               topk_group=2,
                                               renormalize=False,
                                               top_k=2,
                                               num_expert_group=4,
                                               e_score_correction_bias=bias)
        assert tw.shape == (4, 2)
        assert ti.shape == (4, 2)


class TestSelectExpertsWithFusionOps:
    """Cover _select_experts_with_fusion_ops branches."""

    def test_sigmoid_norm_type(self, mock_dist_env, mock_moe_env):
        """Non-softmax scoring_func → norm_type=1."""
        x = torch.randn(4, 8)
        logits = torch.randn(4, 8)
        tw, ti = _select_experts_with_fusion_ops(hidden_states=x,
                                                 router_logits=logits,
                                                 top_k=2,
                                                 use_grouped_topk=False,
                                                 renormalize=False,
                                                 e_score_correction_bias=None,
                                                 topk_group=2,
                                                 num_expert_group=4,
                                                 scoring_func="sigmoid",
                                                 global_num_experts=8)
        assert tw.shape == (4, 2)

    def test_softmax_with_renormalize(self, mock_dist_env, mock_moe_env):
        x = torch.randn(4, 8)
        logits = torch.randn(4, 8)
        tw, ti = _select_experts_with_fusion_ops(hidden_states=x,
                                                 router_logits=logits,
                                                 top_k=2,
                                                 use_grouped_topk=False,
                                                 renormalize=True,
                                                 e_score_correction_bias=None,
                                                 topk_group=None,
                                                 num_expert_group=None,
                                                 scoring_func="softmax",
                                                 global_num_experts=8)
        assert tw.shape == (4, 2)

    def test_with_e_score_correction_bias_dtype_cast(self, mock_dist_env,
                                                     mock_moe_env):
        """e_score_correction_bias with different dtype triggers cast."""
        x = torch.randn(4, 8)
        logits = torch.randn(4, 8, dtype=torch.float32)
        bias = torch.randn(8, dtype=torch.float16)  # different dtype
        tw, ti = _select_experts_with_fusion_ops(hidden_states=x,
                                                 router_logits=logits,
                                                 top_k=2,
                                                 use_grouped_topk=False,
                                                 renormalize=True,
                                                 e_score_correction_bias=bias,
                                                 topk_group=None,
                                                 num_expert_group=None,
                                                 scoring_func="softmax",
                                                 global_num_experts=8)
        assert tw.shape == (4, 2)


# ──────────────────────────────────────────────────────────────────
# 2. moe_mlp coverage
# ──────────────────────────────────────────────────────────────────
class TestCumsumGroupListExtra(TestBase):
    """Additional cumsum_group_list coverage."""

    def test_invalid_group_list_type_raises(self):
        group_list = torch.tensor([1, 2, 3])
        with self.assertRaises(ValueError):
            cumsum_group_list(group_list, group_list_type=3)

    def test_group_list_type_negative_raises(self):
        group_list = torch.tensor([1, 2, 3])
        with self.assertRaises(ValueError):
            cumsum_group_list(group_list, group_list_type=-1)


class TestUnifiedApplyMLPExtra(TestBase):
    """Additional unified_apply_mlp coverage: need_trans=False path."""

    @patch('vllm_ascend.ops.fused_moe.moe_mlp.is_310p')
    @patch('torch_npu.npu_grouped_matmul')
    @patch('torch_npu.npu_swiglu')
    @patch('torch_npu.npu_dynamic_quant')
    def test_unquant_need_trans_false(self, mock_npu_dynamic_quant,
                                      mock_npu_swiglu, mock_npu_grouped_matmul,
                                      mock_is_310p):
        """unquant_apply_mlp with need_trans=False (no transpose)."""
        mock_is_310p.return_value = False
        mock_npu_grouped_matmul.side_effect = [[
            torch.randn(10, 40, dtype=torch.float16)
        ], [torch.randn(10, 20, dtype=torch.float16)]]
        mock_npu_swiglu.return_value = torch.randn(10, 40, dtype=torch.float16)

        hidden_states = torch.randn(10, 20, dtype=torch.float16)
        # w1 and w2 already in correct shape (no transpose needed)
        w1 = torch.randn(5, 40, 20, dtype=torch.float16)
        w2 = torch.randn(5, 20, 40, dtype=torch.float16)
        group_list = torch.tensor([2, 4, 6, 8, 10], dtype=torch.int64)

        result = unified_apply_mlp(hidden_states=hidden_states,
                                   w1=w1,
                                   w1_scale=None,
                                   w2=w2,
                                   w2_scale=None,
                                   group_list=group_list,
                                   dynamic_scale=None,
                                   group_list_type=1,
                                   w1_scale_bias=None,
                                   w2_scale_bias=None,
                                   topk_scales=None,
                                   with_quant=False,
                                   need_trans=False)

        self.assertEqual(result.shape, hidden_states.shape)
        # Verify no transpose calls happened
        call_args = mock_npu_grouped_matmul.call_args_list[0]
        # w1 should be passed as-is (not transposed)
        self.assertEqual(
            call_args.kwargs.get('weight', [None])[0].shape, w1.shape)


# ──────────────────────────────────────────────────────────────────
# 3. fused_moe.py coverage
# ──────────────────────────────────────────────────────────────────
class TestAscendFusedMoEHelpers:
    """Cover helper methods on AscendFusedMoE."""

    def test_get_quant_type_none(self, mock_dist_env):
        """_get_quant_type returns NONE when no quant_method attr."""
        with patch.object(AscendFusedMoE, "__init__",
                          lambda self, *a, **kw: None):
            moe = AscendFusedMoE.__new__(AscendFusedMoE)
            moe.quant_method = MagicMock(spec=[])  # no quant_method attr
            assert moe._get_quant_type() == QuantType.NONE

    def test_get_quant_type_none_when_quant_method_is_none(
            self, mock_dist_env):
        with patch.object(AscendFusedMoE, "__init__",
                          lambda self, *a, **kw: None):
            moe = AscendFusedMoE.__new__(AscendFusedMoE)
            moe.quant_method = MagicMock()
            moe.quant_method.quant_method = None
            assert moe._get_quant_type() == QuantType.NONE

    @patch(
        'vllm_ascend.ops.fused_moe.fused_moe.AscendW8A8DynamicFusedMoEMethod')
    def test_get_quant_type_w8a8(self, mock_w8a8_cls, mock_dist_env):
        with patch.object(AscendFusedMoE, "__init__",
                          lambda self, *a, **kw: None):
            moe = AscendFusedMoE.__new__(AscendFusedMoE)
            mock_method = MagicMock()
            mock_method.__class__ = mock_w8a8_cls
            moe.quant_method = MagicMock()
            moe.quant_method.quant_method = mock_method
            # isinstance check needs special handling
            with patch('vllm_ascend.ops.fused_moe.fused_moe.isinstance',
                       side_effect=lambda obj, cls: cls == mock_w8a8_cls):
                pass
            # Direct check: method is of the right type
            from vllm_ascend.quantization.w8a8_dynamic import \
                AscendW8A8DynamicFusedMoEMethod
            moe.quant_method.quant_method = MagicMock(
                spec=AscendW8A8DynamicFusedMoEMethod)
            result = moe._get_quant_type()
            assert result == QuantType.W8A8

    @patch(
        'vllm_ascend.ops.fused_moe.fused_moe.AscendW4A8DynamicFusedMoEMethod')
    def test_get_quant_type_w4a8(self, mock_w4a8_cls, mock_dist_env):
        with patch.object(AscendFusedMoE, "__init__",
                          lambda self, *a, **kw: None):
            moe = AscendFusedMoE.__new__(AscendFusedMoE)
            from vllm_ascend.quantization.w4a8_dynamic import \
                AscendW4A8DynamicFusedMoEMethod
            moe.quant_method = MagicMock()
            moe.quant_method.quant_method = MagicMock(
                spec=AscendW4A8DynamicFusedMoEMethod)
            result = moe._get_quant_type()
            assert result == QuantType.W4A8

    def test_update_expert_map(self, mock_dist_env):
        with patch.object(AscendFusedMoE, "__init__",
                          lambda self, *a, **kw: None):
            moe = AscendFusedMoE.__new__(AscendFusedMoE)
            moe.expert_map = None
            new_map = torch.tensor([0, 1, 2, -1])
            moe.update_expert_map(new_map)
            assert torch.equal(moe.expert_map, new_map)

    def test_get_map(self, mock_dist_env):
        with patch.object(AscendFusedMoE, "__init__",
                          lambda self, *a, **kw: None):
            moe = AscendFusedMoE.__new__(AscendFusedMoE)
            expected = torch.tensor([0, 1, -1])
            moe.expert_map = expected
            assert torch.equal(moe.get_map(), expected)

    def test_clear_moe_load(self, mock_dist_env):
        with patch.object(AscendFusedMoE, "__init__",
                          lambda self, *a, **kw: None):
            moe = AscendFusedMoE.__new__(AscendFusedMoE)
            moe.moe_load = torch.tensor([10, 20, 30])
            moe.clear_moe_load()
            assert torch.equal(moe.moe_load, torch.zeros(3, dtype=torch.int64))

    def test_clear_moe_load_none(self, mock_dist_env):
        with patch.object(AscendFusedMoE, "__init__",
                          lambda self, *a, **kw: None):
            moe = AscendFusedMoE.__new__(AscendFusedMoE)
            moe.moe_load = None
            moe.clear_moe_load()  # should not raise

    def test_transpose_weight_no_transpose(self, mock_dist_env):
        """When shapes are compatible, no transpose occurs."""
        with patch.object(AscendFusedMoE, "__init__",
                          lambda self, *a, **kw: None):
            moe = AscendFusedMoE.__new__(AscendFusedMoE)
            loaded = torch.randn(64, 8)
            expert = torch.randn(64, 8)
            result_w, result_dim = moe.transpose_weight(loaded, expert, 0)
            assert result_dim == 0
            assert torch.equal(result_w, loaded)

    def test_transpose_weight_needs_transpose(self, mock_dist_env):
        """When both dimensions mismatch, transpose and flip shard_dim."""
        with patch.object(AscendFusedMoE, "__init__",
                          lambda self, *a, **kw: None):
            moe = AscendFusedMoE.__new__(AscendFusedMoE)
            loaded = torch.randn(32, 64)
            expert = torch.randn(128, 16)
            result_w, result_dim = moe.transpose_weight(loaded, expert, 0)
            assert result_dim == 1  # flipped
            assert result_w.shape == (64, 32)  # transposed


class TestProcessWeightsAfterLoading(TestBase):
    """Cover AscendUnquantizedFusedMoEMethod.process_weights_after_loading."""

    @patch('vllm_ascend.ops.fused_moe.fused_moe.get_current_vllm_config')
    @patch('vllm_ascend.ops.fused_moe.fused_moe.get_ascend_config')
    @patch('vllm_ascend.ops.fused_moe.fused_moe.is_310p', return_value=True)
    @patch('vllm_ascend.ops.fused_moe.fused_moe.is_enable_nz',
           return_value=False)
    def test_process_weights_transpose(self, mock_nz, mock_310p,
                                       mock_ascend_cfg, mock_vllm_cfg):
        mock_vllm_cfg.return_value = MagicMock(
            compilation_config=MagicMock(mode=0),
            model_config=MagicMock(enforce_eager=True))
        mock_ascend_cfg.return_value = MagicMock(
            torchair_graph_config=MagicMock(enabled=False), dynamic_eplb=False)

        method = AscendUnquantizedFusedMoEMethod.__new__(
            AscendUnquantizedFusedMoEMethod)
        method.transpose = True
        method.moe = MagicMock()
        method.moe.moe_parallel_config = MagicMock(ep_size=1)
        method.moe.num_experts = 4

        layer = MagicMock()
        layer.w13_weight = torch.nn.Parameter(torch.randn(4, 8, 16))
        layer.w2_weight = torch.nn.Parameter(torch.randn(4, 16, 8))

        # Mock parent class method
        with patch.object(type(method).__mro__[2],
                          'process_weights_after_loading',
                          return_value=None):
            method._maybe_pad_weight = MagicMock(side_effect=lambda x: x)
            method.process_weights_after_loading(layer)

        # transpose should flip to False after first call
        self.assertFalse(method.transpose)
        # w13 should now be transposed (dim 1 and 2 swapped)
        self.assertEqual(layer.w13_weight.shape, (4, 16, 8))
        self.assertEqual(layer.w2_weight.shape, (4, 8, 16))

    @patch('vllm_ascend.ops.fused_moe.fused_moe.get_current_vllm_config')
    @patch('vllm_ascend.ops.fused_moe.fused_moe.get_ascend_config')
    @patch('vllm_ascend.ops.fused_moe.fused_moe.is_310p', return_value=True)
    @patch('vllm_ascend.ops.fused_moe.fused_moe.is_enable_nz',
           return_value=False)
    def test_process_weights_no_transpose(self, mock_nz, mock_310p,
                                          mock_ascend_cfg, mock_vllm_cfg):
        mock_vllm_cfg.return_value = MagicMock(
            compilation_config=MagicMock(mode=0),
            model_config=MagicMock(enforce_eager=True))
        mock_ascend_cfg.return_value = MagicMock(
            torchair_graph_config=MagicMock(enabled=False), dynamic_eplb=False)

        method = AscendUnquantizedFusedMoEMethod.__new__(
            AscendUnquantizedFusedMoEMethod)
        method.transpose = False  # Already transposed
        method.moe = MagicMock()
        method.moe.moe_parallel_config = MagicMock(ep_size=1)
        method.moe.num_experts = 4

        layer = MagicMock()
        w13_data = torch.randn(4, 8, 16)
        w2_data = torch.randn(4, 16, 8)
        layer.w13_weight = torch.nn.Parameter(w13_data.clone())
        layer.w2_weight = torch.nn.Parameter(w2_data.clone())

        with patch.object(type(method).__mro__[2],
                          'process_weights_after_loading',
                          return_value=None):
            method._maybe_pad_weight = MagicMock(side_effect=lambda x: x)
            method.process_weights_after_loading(layer)

        # No transpose, shapes unchanged
        self.assertEqual(layer.w13_weight.shape, (4, 8, 16))
        self.assertEqual(layer.w2_weight.shape, (4, 16, 8))


# ──────────────────────────────────────────────────────────────────
# 4. moe_comm_method coverage
# ──────────────────────────────────────────────────────────────────
class TestGetMoECommMethod:
    """Cover get_moe_comm_method and setup_moe_comm_method."""

    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_current_vllm_config")
    @patch(
        "vllm_ascend.ops.fused_moe.moe_comm_method.AllGatherCommImpl._get_token_dispatcher",
        return_value=None)
    @patch(
        "vllm_ascend.ops.fused_moe.moe_comm_method.MC2CommImpl._get_token_dispatcher",
        return_value=None)
    @patch(
        "vllm_ascend.ops.fused_moe.moe_comm_method.AlltoAllCommImpl._get_token_dispatcher",
        return_value=None)
    def test_setup_and_get(self, mock_a2a_td, mock_mc2_td, mock_ag_td,
                           mock_cfg):
        mock_cfg.return_value = MagicMock()
        moe_config = MagicMock(spec=FusedMoEConfig)
        moe_config.num_experts = 8
        moe_config.num_local_experts = 2
        moe_config.experts_per_token = 2
        moe_config.num_global_redundant_experts = 0
        moe_config.tp_group = MagicMock()
        moe_config.dp_size = 1
        moe_config.tp_size = 1
        moe_config.ep_size = 1
        moe_config.dp_group = MagicMock()
        moe_config.original_num_experts = 8

        setup_moe_comm_method(moe_config)

        assert get_moe_comm_method(MoECommType.ALLTOALL) is not None
        assert get_moe_comm_method(MoECommType.ALLGATHER) is not None
        assert get_moe_comm_method(MoECommType.MC2) is not None

    def test_get_nonexistent_returns_none(self):
        assert get_moe_comm_method(None) is None


class TestAllGatherCommImplDispatcher:
    """Cover AllGatherCommImpl._get_token_dispatcher PanguProMoE branch."""

    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_current_vllm_config")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_forward_context")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.TokenDispatcherWithMoge")
    @patch(
        "vllm_ascend.ops.fused_moe.moe_comm_method.PrepareAndFinalizeWithAllGather"
    )
    def test_pangu_pro_moe_dispatcher(self, mock_pf, mock_moge_td,
                                      mock_fwd_ctx, mock_cfg):
        mock_vllm_config = MagicMock()
        mock_vllm_config.model_config.hf_config.model_type = "PanguProMoE"
        mock_cfg.return_value = mock_vllm_config

        moe_config = MagicMock(spec=FusedMoEConfig)
        moe_config.experts_per_token = 2
        moe_config.num_experts = 8
        moe_config.num_local_experts = 2
        moe_config.num_global_redundant_experts = 0
        moe_config.dp_size = 1
        moe_config.tp_size = 1
        moe_config.ep_size = 1
        moe_config.tp_group = MagicMock()
        moe_config.dp_group = MagicMock()
        moe_config.original_num_experts = 8

        AllGatherCommImpl(moe_config)
        mock_moge_td.assert_called_once()


class TestMoECommMethodFusedExpertsEplb:
    """Cover fused_experts with dynamic_eplb=True (returns tuple)."""

    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_current_vllm_config")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_forward_context")
    @patch(
        "vllm_ascend.ops.fused_moe.moe_comm_method.PrepareAndFinalizeWithAllGather"
    )
    @patch(
        "vllm_ascend.ops.fused_moe.moe_comm_method.TokenDispatcherWithAllGather"
    )
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.unified_apply_mlp")
    def test_fused_experts_dynamic_eplb(self, mock_mlp, mock_td_cls,
                                        mock_pf_cls, mock_fwd_ctx, mock_cfg):
        mock_cfg.return_value = MagicMock()
        mock_fwd_ctx.return_value = MagicMock(moe_comm_method=MagicMock())

        mock_pf = MagicMock()
        mock_pf_cls.return_value = mock_pf

        mock_td = MagicMock()
        expert_tokens = torch.tensor([2, 2, 2])
        mock_td.token_dispatch.return_value = {
            "hidden_states": torch.randn(6, 8),
            "group_list": expert_tokens,
            "group_list_type": 1,
            "dynamic_scale": None,
            "context_metadata": {}
        }
        mock_td.token_combine.return_value = torch.randn(4, 8)
        mock_td_cls.return_value = mock_td
        mock_mlp.return_value = torch.randn(6, 8)

        moe_config = MagicMock(spec=FusedMoEConfig)
        moe_config.experts_per_token = 2
        moe_config.num_experts = 8
        moe_config.num_local_experts = 2
        moe_config.num_global_redundant_experts = 0
        moe_config.dp_size = 1
        moe_config.tp_size = 1
        moe_config.ep_size = 1
        moe_config.tp_group = MagicMock()
        moe_config.dp_group = MagicMock()
        moe_config.original_num_experts = 8

        comm = AllGatherCommImpl(moe_config)

        hs = torch.randn(4, 8).contiguous()
        w1 = torch.randn(16, 8).contiguous()
        w2 = torch.randn(16, 8).contiguous()
        tw = torch.tensor([[0.5, 0.5], [0.3, 0.7], [0.8, 0.2], [0.6, 0.4]])
        ti = torch.tensor([[0, 1], [1, 2], [2, 0], [1, 1]])

        result = comm.fused_experts(hidden_states=hs,
                                    w1=w1,
                                    w2=w2,
                                    topk_weights=tw,
                                    topk_ids=ti,
                                    dynamic_eplb=True)

        # dynamic_eplb=True returns tuple of (hidden_states, group_list_type, expert_tokens)
        assert isinstance(result, tuple)
        assert len(result) == 3


# ──────────────────────────────────────────────────────────────────
# 5. prepare_finalize coverage
# ──────────────────────────────────────────────────────────────────
class TestPrepareAndFinalizeExtra(unittest.TestCase):
    """Cover additional prepare_finalize branches."""

    def setUp(self):
        self.moe_config = MagicMock(spec=FusedMoEConfig)
        self.moe_config.tp_group = MagicMock()
        self.moe_config.tp_group.device_group = MagicMock()
        self.moe_config.dp_size = 1
        self.moe_config.tp_size = 1
        self.moe_config.ep_size = 1
        self.moe_config.dp_group = MagicMock()
        self.moe_config.original_num_experts = 8

    @patch(
        "vllm_ascend.ops.fused_moe.prepare_finalize.get_tensor_model_parallel_world_size",
        return_value=1)
    @patch(
        "vllm_ascend.ops.fused_moe.prepare_finalize.get_tensor_model_parallel_rank",
        return_value=0)
    @patch("vllm_ascend.ops.fused_moe.prepare_finalize.get_forward_context")
    def test_mc2_with_shared_expert_dp(self, mock_ctx, mock_rank, mock_ws):
        """MC2 prepare with enable_shared_expert_dp=True skips padding."""
        mock_ctx.return_value = MagicMock(mc2_mask=torch.tensor([1, 0, 1]),
                                          padded_num_tokens=4)

        layer = PrepareAndFinalizeWithMC2(self.moe_config)
        hs = torch.randn(3, 8)
        rl = torch.randn(3, 2)

        h_out, r_out, mask, ctx_meta = layer.prepare(
            hs, rl, enable_shared_expert_dp=True)
        # No padding when shared expert dp
        self.assertEqual(h_out.shape[0], 3)

    @patch(
        "vllm_ascend.ops.fused_moe.prepare_finalize.get_tensor_model_parallel_world_size",
        return_value=1)
    @patch(
        "vllm_ascend.ops.fused_moe.prepare_finalize.get_tensor_model_parallel_rank",
        return_value=0)
    @patch("vllm_ascend.ops.fused_moe.prepare_finalize.get_forward_context")
    def test_mc2_with_replace_allreduce(self, mock_ctx, mock_rank, mock_ws):
        """MC2 prepare with replace_allreduce=True skips all processing."""
        mock_ctx.return_value = MagicMock(mc2_mask=torch.tensor([1, 0, 1]),
                                          padded_num_tokens=4)

        layer = PrepareAndFinalizeWithMC2(self.moe_config)
        hs = torch.randn(3, 8)
        rl = torch.randn(3, 2)

        h_out, r_out, mask, ctx_meta = layer.prepare(hs,
                                                     rl,
                                                     replace_allreduce=True)
        # When replace_allreduce, hidden_states are unchanged
        self.assertEqual(h_out.shape[0], 3)

    @patch(
        "vllm_ascend.ops.fused_moe.prepare_finalize.get_tensor_model_parallel_world_size",
        return_value=1)
    @patch(
        "vllm_ascend.ops.fused_moe.prepare_finalize.get_tensor_model_parallel_rank",
        return_value=0)
    def test_all2all_with_shared_expert_dp(self, mock_rank, mock_ws):
        """All2All prepare with enable_shared_expert_dp=True."""
        layer = PrepareAndFinalizeWithAll2All(self.moe_config)
        hs = torch.randn(3, 8)
        rl = torch.randn(3, 2)

        h_out, r_out, _, ctx_meta = layer.prepare(hs,
                                                  rl,
                                                  enable_shared_expert_dp=True)
        # Skips padding when shared expert dp
        self.assertEqual(h_out.shape[0], 3)

    @patch(
        "vllm_ascend.ops.fused_moe.prepare_finalize.get_tensor_model_parallel_world_size",
        return_value=1)
    @patch(
        "vllm_ascend.ops.fused_moe.prepare_finalize.get_tensor_model_parallel_rank",
        return_value=0)
    def test_all2all_finalize_no_unpad(self, mock_rank, mock_ws):
        """All2All finalize when num_tokens == hidden_states size (no unpad)."""
        layer = PrepareAndFinalizeWithAll2All(self.moe_config)
        hs = torch.randn(4, 8)
        rl = torch.randn(4, 2)

        _, _, _, ctx_meta = layer.prepare(hs, rl)
        result = layer.finalize(hs,
                                reduce_results=False,
                                context_metadata=ctx_meta)
        # Should return same size (no unpadding needed)
        self.assertEqual(result.shape[0], 4)

    @patch("vllm_ascend.ops.fused_moe.prepare_finalize.get_dp_group")
    @patch(
        "vllm_ascend.ops.fused_moe.prepare_finalize.tensor_model_parallel_all_reduce"
    )
    @patch("vllm_ascend.ops.fused_moe.prepare_finalize.get_forward_context")
    @patch("vllm_ascend.ops.fused_moe.prepare_finalize.enable_sp",
           return_value=False)
    def test_allgather_dp1_no_gather(self, mock_sp, mock_ctx, mock_reduce,
                                     mock_dp):
        """AllGather with dp_size=1 skips all-gather."""
        mock_ctx.return_value = MagicMock(max_tokens_across_dp=6)
        self.moe_config.dp_size = 1

        layer = PrepareAndFinalizeWithAllGather(self.moe_config)
        hs = torch.randn(3, 8)
        rl = torch.randn(3, 2)

        h_out, r_out, _, ctx_meta = layer.prepare(hs, rl)
        # dp_size=1, no all-gather
        self.assertEqual(h_out.shape[0], 3)

    @patch("vllm_ascend.ops.fused_moe.prepare_finalize.get_dp_group")
    @patch(
        "vllm_ascend.ops.fused_moe.prepare_finalize.tensor_model_parallel_all_reduce"
    )
    @patch("vllm_ascend.ops.fused_moe.prepare_finalize.get_forward_context")
    @patch("vllm_ascend.ops.fused_moe.prepare_finalize.enable_sp",
           return_value=False)
    def test_allgather_finalize_with_tp_reduce(self, mock_sp, mock_ctx,
                                               mock_reduce, mock_dp):
        """AllGather finalize with reduce_results=True and tp>1."""
        mock_ctx.return_value = MagicMock(max_tokens_across_dp=4)
        mock_dp_group = MagicMock()
        mock_dp_group.all_gather = lambda t, d: torch.cat([t, t], dim=d)
        mock_dp_group.reduce_scatter = lambda t, d: t[:2]
        mock_dp.return_value = mock_dp_group

        self.moe_config.dp_size = 2
        self.moe_config.tp_size = 2
        self.moe_config.dp_group = mock_dp_group

        layer = PrepareAndFinalizeWithAllGather(self.moe_config)
        hs = torch.randn(4, 8)
        rl = torch.randn(4, 2)

        h_out, _, _, _ = layer.prepare(hs, rl)

        mock_reduce.return_value = torch.randn(2, 8)
        layer.finalize(h_out, reduce_results=True)
        mock_reduce.assert_called_once()


class TestQuantTypeEnum(unittest.TestCase):
    """Ensure QuantType enum values are accessible."""

    def test_quant_type_values(self):
        self.assertEqual(QuantType.NONE.value, 0)
        self.assertEqual(QuantType.W8A8.value, 1)
        self.assertEqual(QuantType.W4A8.value, 2)


# ──────────────────────────────────────────────────────────────────
# 6. AscendSharedFusedMoE coverage
# ──────────────────────────────────────────────────────────────────
class TestAscendSharedFusedMoE:
    """Cover AscendSharedFusedMoE properties."""

    def test_gate_property_overlapped(self, mock_dist_env):
        with patch.object(AscendSharedFusedMoE, "__init__",
                          lambda self, *a, **kw: None):
            moe = AscendSharedFusedMoE.__new__(AscendSharedFusedMoE)
            mock_gate = MagicMock()
            moe._gate = mock_gate
            moe.use_overlapped = True
            assert moe.gate is mock_gate

    def test_gate_property_not_overlapped(self, mock_dist_env):
        with patch.object(AscendSharedFusedMoE, "__init__",
                          lambda self, *a, **kw: None):
            moe = AscendSharedFusedMoE.__new__(AscendSharedFusedMoE)
            mock_gate = MagicMock()
            moe._gate = mock_gate
            moe.use_overlapped = False
            assert moe.gate is None

    def test_is_internal_router(self, mock_dist_env):
        with patch.object(AscendSharedFusedMoE, "__init__",
                          lambda self, *a, **kw: None):
            moe = AscendSharedFusedMoE.__new__(AscendSharedFusedMoE)
            assert moe.is_internal_router is False


# ──────────────────────────────────────────────────────────────────
# 7. comm_utils coverage
# ──────────────────────────────────────────────────────────────────
class TestCommUtils(unittest.TestCase):
    """Cover comm_utils functions."""

    @patch("torch.distributed.get_world_size", return_value=1)
    def test_gather_along_first_dim_single_gpu(self, mock_ws):
        from vllm_ascend.ops.fused_moe.comm_utils import \
            _gather_along_first_dim
        inp = torch.randn(4, 8)
        result = _gather_along_first_dim(inp, group=None)
        self.assertTrue(torch.equal(result, inp))

    @patch("torch.distributed.all_to_all_single")
    def test_async_all_to_all_equal_split(self, mock_a2a):
        from vllm_ascend.ops.fused_moe.comm_utils import async_all_to_all
        mock_handle = MagicMock()
        mock_a2a.return_value = mock_handle

        inp = torch.randn(8, 4)
        _, out, handle = async_all_to_all(inp,
                                          output_split_sizes=None,
                                          input_split_sizes=None,
                                          group=None)
        self.assertEqual(out.shape, inp.shape)
        self.assertIs(handle, mock_handle)

    @patch("torch.distributed.all_to_all_single")
    def test_async_all_to_all_unequal_split(self, mock_a2a):
        from vllm_ascend.ops.fused_moe.comm_utils import async_all_to_all
        mock_handle = MagicMock()
        mock_a2a.return_value = mock_handle

        inp = torch.randn(8, 4)
        out_splits = [3, 5]
        _, out, handle = async_all_to_all(inp,
                                          output_split_sizes=out_splits,
                                          input_split_sizes=[4, 4],
                                          group=None)
        self.assertEqual(out.shape[0], 8)  # sum of out_splits
        self.assertIs(handle, mock_handle)


# ──────────────────────────────────────────────────────────────────
# 8. load weight coverage (extended)
# ──────────────────────────────────────────────────────────────────
class TestLoadWeightExtra(TestBase):
    """Extended load weight tests."""

    def test_load_w13_with_load_full(self):
        """_load_w13 with load_full=True skips narrow."""
        with patch.object(AscendFusedMoE, "__init__",
                          lambda self, *a, **kw: None):
            moe = AscendFusedMoE(num_experts=4, top_k=2, hidden_size=8)
            expert_data = torch.randn(128, 8)
            loaded_weight = torch.randn(128, 4)
            moe._load_w13(expert_data,
                          1,
                          "w1",
                          loaded_weight,
                          0,
                          load_full=True)

    def test_load_w2_with_load_full(self):
        """_load_w2 with load_full=True skips narrow."""
        with patch.object(AscendFusedMoE, "__init__",
                          lambda self, *a, **kw: None):
            moe = AscendFusedMoE(num_experts=4, top_k=2, hidden_size=8)
            expert_data = torch.randn(128, 4)
            loaded_weight = torch.randn(128, 4)
            moe._load_w2(expert_data, 1, loaded_weight, 0, load_full=True)
