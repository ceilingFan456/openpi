import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.enable_aux_2d = config.enable_aux_2d
        self.aux_2d_weight = config.aux_2d_weight
        self.policy_weight = config.policy_weight
        self.aux_horizon = int(config.aux_horizon)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        ## keep this for language input for the 2d-auxiliary head mlp to do film. 
        self._paligemma_width = paligemma_config.width
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
        # Small FiLM-conditioned head for auxiliary 2D waypoint prediction.
        if config.enable_aux_2d:
            self.aux_lang_proj = nnx.Linear(self._paligemma_width, config.aux_mlp_dim, rngs=rngs)
            self.aux_vision_proj = nnx.Linear(self._paligemma_width, config.aux_mlp_dim, rngs=rngs)
            self.aux_film_gamma = nnx.Linear(config.aux_mlp_dim, config.aux_mlp_dim, rngs=rngs)
            self.aux_film_beta = nnx.Linear(config.aux_mlp_dim, config.aux_mlp_dim, rngs=rngs)
            self.aux_head_hidden = nnx.Linear(config.aux_mlp_dim, config.aux_mlp_dim, rngs=rngs)
            self.aux_head_out = nnx.Linear(config.aux_mlp_dim, 2 * self.aux_horizon, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    ## new function to generate auxiliary 2d predictions and do film before the auxiliary head mlp.
    ## pooled vision features from image tokens.
    ## pooled language embeddings from prompt tokens. 
    ## film modulation + mlp head -> [b, aux_horizon, 2] predicted 2d waypoints. 
    def _compute_aux_predictions(
        self, observation: _model.Observation, *, train: bool
    ) -> tuple[at.Float[at.Array, "b h 2"], at.Bool[at.Array, " b"]]:
        """Predicts future 2D waypoints from vision features, FiLM-conditioned on language."""
        vision_features = []
        vision_feature_masks = []
        for name in observation.images:
            image_tokens, _ = self.PaliGemma.img(observation.images[name], train=train)
            # Pool token features per image stream, then combine streams.
            vision_features.append(jnp.mean(image_tokens, axis=1))
            vision_feature_masks.append(observation.image_masks[name].astype(jnp.float32)[:, None])

        # Weighted mean over image streams with image validity masks.
        stacked_vision = jnp.stack(vision_features, axis=1)
        stacked_vision_mask = jnp.stack(vision_feature_masks, axis=1)
        pooled_vision = jnp.sum(stacked_vision * stacked_vision_mask, axis=1) / jnp.clip(
            jnp.sum(stacked_vision_mask, axis=1), a_min=1.0
        )

        if observation.tokenized_prompt is not None and observation.tokenized_prompt_mask is not None:
            lang_tokens = self.PaliGemma.llm(observation.tokenized_prompt, method="embed")
            lang_mask = observation.tokenized_prompt_mask.astype(jnp.float32)[..., None]
            pooled_lang = jnp.sum(lang_tokens * lang_mask, axis=1) / jnp.clip(jnp.sum(lang_mask, axis=1), a_min=1.0)
        else:
            pooled_lang = jnp.zeros((pooled_vision.shape[0], self._paligemma_width), dtype=pooled_vision.dtype)

        # FiLM modulation: modulate vision features with language context.
        aux_vis = self.aux_vision_proj(pooled_vision)
        aux_lang = self.aux_lang_proj(pooled_lang)
        gamma = self.aux_film_gamma(aux_lang)
        beta = self.aux_film_beta(aux_lang)
        fused = aux_vis * (1.0 + gamma) + beta
        fused = nnx.swish(fused)
        fused = self.aux_head_hidden(fused)
        fused = nnx.swish(fused)
        pred = self.aux_head_out(fused)
        pred = pred.reshape(pred.shape[0], self.aux_horizon, 2)

        if observation.use_auxiliary is None:
            use_aux = jnp.zeros((pred.shape[0],), dtype=bool)
        else:
            use_aux = observation.use_auxiliary.astype(bool)
        return pred, use_aux

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        policy_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)

        # Gate policy loss on a per-sample basis. If no gate is provided, keep legacy behavior (all enabled).
        ## so use the boolean `use_policy` field in the observation as a filter for the policy loss. 
        ## so it is operated on a per-sample basis not per-batch basis. 
        if observation.use_policy is None:
            policy_gate = jnp.ones((policy_loss.shape[0],), dtype=policy_loss.dtype)
        else:
            policy_gate = observation.use_policy.astype(policy_loss.dtype)
        total_loss = self.policy_weight * (policy_loss * policy_gate[:, None])

        if self.enable_aux_2d and observation.aux_keypoints_2d is not None:
            pred_2d, aux_gate = self._compute_aux_predictions(observation, train=train)
            target_2d = observation.aux_keypoints_2d
            ## sanity check to make sure the dataset matches the model config.
            if target_2d.shape[1] != self.aux_horizon:
                raise ValueError(
                    f"aux_keypoints_2d has horizon {target_2d.shape[1]} but model aux_horizon is {self.aux_horizon}"
                )

            # If no waypoint mask is provided, treat all waypoints as valid.
            if observation.aux_keypoints_mask is None:
                point_mask = jnp.ones((target_2d.shape[0], target_2d.shape[1]), dtype=target_2d.dtype)
            else:
                point_mask = observation.aux_keypoints_mask.astype(target_2d.dtype)

            ## point mask refers to weather a point is considered for loss,
            ## aux_gate refers to whether a sample is considered for loss.
            aux_sq_err = jnp.mean(jnp.square(pred_2d - target_2d), axis=-1) ## mean is along action horizon dimension. 
            aux_loss_per_sample = jnp.sum(aux_sq_err * point_mask, axis=-1) / jnp.clip(jnp.sum(point_mask, axis=-1), 1.0)
            aux_loss_per_sample = aux_loss_per_sample * aux_gate.astype(aux_loss_per_sample.dtype)
            total_loss = total_loss + self.aux_2d_weight * aux_loss_per_sample[:, None]

        return total_loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
