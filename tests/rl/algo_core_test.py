# Copyright 2026 Google LLC
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

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from tunix.rl import algo_core


class AlgoCoreTest(absltest.TestCase):

  def test_compute_rloo_advantages(self):
    rewards = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    advantages = algo_core.compute_rloo_advantages(rewards, num_generations=3)
    expected_value = jnp.array([-1.5, 0.0, 1.5, -1.5, 0.0, 1.5])
    np.testing.assert_allclose(advantages, expected_value)

  def test_compute_rloo_advantages_low_generations(self):
    rewards = jnp.array([1.0, 2.0])
    advantages = algo_core.compute_rloo_advantages(rewards, num_generations=1)
    np.testing.assert_allclose(advantages, jnp.zeros_like(rewards))

  def test_grpo_compute_advantages(self):
    prev_val = jax.config.jax_threefry_partitionable
    self.addCleanup(jax.config.update, 'jax_threefry_partitionable', prev_val)
    jax.config.update('jax_threefry_partitionable', False)
    self.assertFalse(jax.config.jax_threefry_partitionable)

    rng = jax.random.PRNGKey(0)
    rewards = jax.random.uniform(rng, shape=(1, 6))
    advantages = algo_core.compute_advantages(rewards, num_generations=3)
    expected_value = jnp.array(
        [[0.307498, -1.117636, 0.810138, 1.094526, -0.228671, -0.865855]]
    )
    np.testing.assert_allclose(advantages, expected_value, rtol=1e-3, atol=1e-3)

  def test_grpo_loss_fn_packed_equals_unpacked(self):
    # P3.4 gate: grpo_loss_fn gives the SAME primary loss whether two sequences
    # are packed into one row (segment_ids set) or one-per-row (segment_ids
    # None). Proves segment_ids/num_segments are threaded into the loss
    # aggregation and the gspo-token per-segment pooling. old_per_token_logps is
    # None (is_ratio == 1), so the model output cancels and this isolates the
    # aggregation wiring: sequence-mean-token-mean over A (adv 1.5, 3 tokens) and
    # B (adv 3.0, 1 token) = (-1.5 + -3.0) / 2 = -2.25; a broken per-row
    # aggregation would instead give -1.875.
    from types import SimpleNamespace  # pylint: disable=g-import-not-at-top
    from flax import nnx  # pylint: disable=g-import-not-at-top
    from tunix.rl import common  # pylint: disable=g-import-not-at-top

    class _SegAwareToy(nnx.Module):
      """Tiny model whose attention is confined to same-segment positions."""

      def __init__(self, *, vocab, dim, rngs):
        self.emb = nnx.Embed(vocab, dim, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=2,
            in_features=dim,
            qkv_features=dim,
            use_bias=False,
            decode=False,
            rngs=rngs,
        )
        self.head = nnx.Linear(dim, vocab, rngs=rngs)

      def __call__(
          self,
          x,
          segment_ids=None,
          positions=None,
          cache=None,
          attention_mask=None,
      ):
        h = self.emb(x)
        if segment_ids is not None:
          same_seg = segment_ids[:, :, None] == segment_ids[:, None, :]
          h = self.attn(h, mask=same_seg[:, None, :, :]) + h
        else:
          h = self.attn(h) + h
        return self.head(h), cache

    model = _SegAwareToy(vocab=16, dim=8, rngs=nnx.Rngs(0))
    packed = common.TrainExample(
        prompt_ids=jnp.zeros((1, 0), jnp.int32),
        prompt_mask=jnp.zeros((1, 0), jnp.int32),
        completion_ids=jnp.array([[3, 4, 5, 6]], jnp.int32),
        completion_mask=jnp.array([[1, 1, 1, 1]], jnp.float32),
        advantages=jnp.array([[1.5, 1.5, 1.5, 3.0]], jnp.float32),
        ref_per_token_logps=None,
        old_per_token_logps=None,
        segment_ids=jnp.array([[1, 1, 1, 2]], jnp.int32),
        segment_positions=jnp.array([[0, 1, 2, 0]], jnp.int32),
        num_segments=3,
    )
    unpacked = common.TrainExample(
        prompt_ids=jnp.array([[7], [7]], jnp.int32),
        prompt_mask=jnp.array([[1], [1]], jnp.int32),
        completion_ids=jnp.array([[3, 4, 5], [6, 0, 0]], jnp.int32),
        completion_mask=jnp.array([[1, 1, 1], [1, 0, 0]], jnp.float32),
        advantages=jnp.array([1.5, 3.0], jnp.float32),
        ref_per_token_logps=None,
        old_per_token_logps=None,
        segment_ids=None,
        segment_positions=None,
        num_segments=None,
    )
    for loss_algo in ('grpo', 'gspo-token'):
      cfg = SimpleNamespace(
          beta=0.0,
          epsilon=0.2,
          epsilon_high=0.2,
          epsilon_c=None,
          loss_algo=loss_algo,
          loss_agg_mode='sequence-mean-token-mean',
          temperature=1.0,
          kl_loss_mode='low_var_kl',
          kl_clamp_value=None,
          force_compute_kl=False,
      )
      lp = float(
          algo_core.grpo_loss_fn(
              model, packed, cfg, pad_id=0, eos_id=-1
          ).primary_loss.compute()
      )
      lu = float(
          algo_core.grpo_loss_fn(
              model, unpacked, cfg, pad_id=0, eos_id=-1
          ).primary_loss.compute()
      )
      with self.subTest(loss_algo=loss_algo):
        np.testing.assert_allclose(lp, lu, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(lp, -2.25, rtol=1e-4, atol=1e-4)


class SamplerIsLengthScalingTest(absltest.TestCase):
  """The sampler-vs-trainer offset length-scaling diagnostic.

  The statistic has to separate two hypotheses that look identical in a batch
  average but imply opposite things at production sequence length: iid token
  noise (the per-sequence offset shrinks as 1/sqrt(T)) versus a systematic
  within-sequence bias (it does not shrink at all).
  """

  _EDGES = (256, 512, 1024)
  _LENGTHS = (128, 384, 768, 2048)  # one per bucket, incl. the open-ended one
  _N_PER_LENGTH = 400
  _TOKEN_SIGMA = 0.018  # per-token log-ratio scatter, as measured on qwen3-4b

  def _synthesize(self, per_sequence_offset_sigma):
    """A batch of mixed-length sequences under a known hypothesis.

    Args:
      per_sequence_offset_sigma: Scale of a constant offset added to every token
        of a sequence. 0.0 gives pure iid token noise; a positive value adds the
        systematic component that sqrt(T) averaging cannot remove.

    Returns:
      (log_is, completion_mask), right-padded to the longest length.
    """
    rng = np.random.default_rng(0)
    t_max = max(self._LENGTHS)
    log_is = np.zeros(
        (len(self._LENGTHS) * self._N_PER_LENGTH, t_max), dtype=np.float32
    )
    mask = np.zeros_like(log_is)
    for i, length in enumerate(self._LENGTHS):
      rows = slice(i * self._N_PER_LENGTH, (i + 1) * self._N_PER_LENGTH)
      tokens = rng.normal(
          0.0, self._TOKEN_SIGMA, size=(self._N_PER_LENGTH, length)
      )
      if per_sequence_offset_sigma:
        tokens += rng.normal(
            0.0, per_sequence_offset_sigma, size=(self._N_PER_LENGTH, 1)
        )
      log_is[rows, :length] = tokens
      mask[rows, :length] = 1.0
    return jnp.asarray(log_is), jnp.asarray(mask)

  def _bucket_sums(self, per_sequence_offset_sigma, overlong=None):
    log_is, completion_mask = self._synthesize(per_sequence_offset_sigma)
    seq_log_mean = (log_is * completion_mask).sum(axis=-1) / (
        completion_mask.sum(axis=-1) + 1e-8
    )
    return algo_core.sampler_is_length_bucket_sums(
        log_is,
        seq_log_mean,
        completion_mask,
        jnp.ones_like(seq_log_mean),
        self._EDGES,
        overlong=overlong,
    )

  def _excess_per_bucket(self, per_sequence_offset_sigma, status='complete'):
    """Runs the diagnostic and applies its documented offline formulas."""
    sums = self._bucket_sums(per_sequence_offset_sigma)
    excess = []
    for name in algo_core.sampler_is_length_bucket_names(self._EDGES):
      prefix = f'{name}/{status}'
      count = float(sums[f'{prefix}/count'])
      observed_rms = np.sqrt(float(sums[f'{prefix}/logmean_sq_sum']) / count)
      iid_rms = np.sqrt(float(sums[f'{prefix}/iid_var_sum']) / count)
      excess.append(observed_rms / iid_rms)
    return excess

  def test_bucket_names(self):
    self.assertEqual(
        algo_core.sampler_is_length_bucket_names((256, 1024)),
        ['le256', 'le1024', 'gt1024'],
    )
    self.assertEqual(algo_core.sampler_is_length_bucket_names(None), [])
    self.assertEqual(algo_core.sampler_is_length_bucket_names(()), [])

  def test_buckets_partition_the_batch_by_length(self):
    sums = self._bucket_sums(0.0)
    names = algo_core.sampler_is_length_bucket_names(self._EDGES)
    # Every sequence lands in exactly one bucket, and each bucket holds the
    # length it was built from. With no truncation verdict supplied, all of
    # them report as complete.
    counts = [float(sums[f'{n}/complete/count']) for n in names]
    self.assertEqual(counts, [float(self._N_PER_LENGTH)] * len(names))
    self.assertEqual(
        [float(sums[f'{n}/truncated/count']) for n in names], [0.0] * len(names)
    )
    for name, length in zip(names, self._LENGTHS):
      mean_length = (
          float(sums[f'{name}/complete/len_sum']) / self._N_PER_LENGTH
      )
      self.assertAlmostEqual(mean_length, length, places=3)

  def test_completion_status_split_is_disjoint_and_additive(self):
    # Half the sequences marked truncated: the two series must partition the
    # bucket, so that summing them recovers the unsplit total.
    n_rows = len(self._LENGTHS) * self._N_PER_LENGTH
    overlong = jnp.asarray(np.tile([0.0, 1.0], n_rows // 2), dtype=jnp.float32)
    sums = self._bucket_sums(0.0, overlong=overlong)
    unsplit = self._bucket_sums(0.0)
    for name in algo_core.sampler_is_length_bucket_names(self._EDGES):
      complete = float(sums[f'{name}/complete/count'])
      truncated = float(sums[f'{name}/truncated/count'])
      self.assertEqual(complete, self._N_PER_LENGTH / 2)
      self.assertEqual(truncated, self._N_PER_LENGTH / 2)
      for metric in algo_core.SAMPLER_IS_LENGTH_BUCKET_METRICS:
        np.testing.assert_allclose(
            float(sums[f'{name}/complete/{metric}'])
            + float(sums[f'{name}/truncated/{metric}']),
            float(unsplit[f'{name}/complete/{metric}']),
            rtol=1e-5,
            err_msg=f'{name}/{metric} is not additive across the split',
        )

  def test_metric_names_cover_what_the_sums_emit(self):
    # The learner registers aggregators from the name list, so it must match
    # the keys the loss actually produces, exactly.
    self.assertCountEqual(
        algo_core.sampler_is_length_bucket_metric_names(self._EDGES),
        list(self._bucket_sums(0.0).keys()),
    )
    self.assertEmpty(algo_core.sampler_is_length_bucket_metric_names(None))

  def test_iid_noise_gives_flat_unit_excess(self):
    # Pure iid tokens: the observed per-sequence spread is exactly what
    # 1/sqrt(T) averaging predicts, at every length.
    excess = self._excess_per_bucket(0.0)
    for name, value in zip(
        algo_core.sampler_is_length_bucket_names(self._EDGES), excess
    ):
      self.assertBetween(value, 0.9, 1.15, msg=f'bucket {name}')

  def test_systematic_offset_gives_excess_growing_with_length(self):
    # A per-sequence constant offset survives averaging, so the observed spread
    # stays flat in T while the iid prediction keeps falling -- their ratio must
    # therefore grow with bucket length. This is the signature to look for.
    excess = self._excess_per_bucket(per_sequence_offset_sigma=0.005)
    self.assertTrue(
        all(a < b for a, b in zip(excess, excess[1:])),
        msg=f'excess should increase with bucket length, got {excess}',
    )
    # 16x length range between the first and last bucket, so ~4x in sqrt(T).
    self.assertGreater(excess[-1] / excess[0], 3.0)


if __name__ == '__main__':
  absltest.main()
