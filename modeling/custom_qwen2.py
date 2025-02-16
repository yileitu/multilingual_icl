from torch import nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2Config, Qwen2DecoderLayer, Qwen2ForCausalLM, Qwen2MLP, \
	Qwen2Model


class MyQwen2MLP(Qwen2MLP):
	def __init__(self, config: Qwen2Config, layer_idx: int):
		super().__init__(config)
		self.config = config
		self.layer_idx = layer_idx

	def forward(self, hidden_state):
		# Get custom attribute in case they are not initialized
		deactivate_neurons = getattr(self.config, 'deactivate_neurons', False)
		count_act = getattr(self.config, 'count_act', False)

		if deactivate_neurons is False:
			if count_act is False:
				# Original implementation
				down_proj = self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))
			else:
				# Count activation over zero
				"""
				hidden_state: batch_size x seq_len x hidden_size
				act: batch_size x seq_len x intermediate_size
				"""
				act = self.act_fn(self.gate_proj(hidden_state))
				self.config.act_over_zero[self.layer_idx, :] += (act > 0).sum(
					dim=(0, 1)
					)  # .sum(dim=(0, 1)) sums over batch_size and seq_len
				down_proj = self.down_proj(act * self.up_proj(hidden_state))
		else:
			# Deactivate neurons
			act = self.act_fn(self.gate_proj(hidden_state))
			layer_mask = self.config.act_mask[self.layer_idx].to(self.config.device)
			act.index_fill_(dim=2, index=layer_mask, value=0.0)  # Deactivate language-specific neurons
			down_proj = self.down_proj(act * self.up_proj(hidden_state))

		return down_proj


class MyQwen2DecoderLayer(Qwen2DecoderLayer):
	def __init__(self, config: Qwen2Config, layer_idx: int):
		super().__init__(config, layer_idx)
		self.mlp = MyQwen2MLP(config, layer_idx)


class MyQwen2Model(Qwen2Model):
	def __init__(self, config: Qwen2Config):
		super().__init__(config)
		self.layers = nn.ModuleList(
			[MyQwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
			)


class MyQwen2ForCausalLM(Qwen2ForCausalLM):
	def __init__(self, config: Qwen2Config):
		super().__init__(config)
		self.model = MyQwen2Model(config)
