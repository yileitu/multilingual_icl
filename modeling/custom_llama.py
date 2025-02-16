from torch import nn
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaDecoderLayer, LlamaForCausalLM, LlamaMLP, \
	LlamaModel


class MyLlamaMLP(LlamaMLP):
	def __init__(self, config: LlamaConfig, layer_idx: int):
		super().__init__(config)
		self.layer_idx = layer_idx

	def forward(self, hidden_state):
		# Note: Ignored self.config.pretraining_tp for now. It is always set to be 1 throughout the experiments.
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


class MyLlamaDecoderLayer(LlamaDecoderLayer):
	def __init__(self, config: LlamaConfig, layer_idx: int):
		super().__init__(config, layer_idx)
		self.mlp = MyLlamaMLP(config, layer_idx)


class MyLlamaModel(LlamaModel):
	def __init__(self, config: LlamaConfig):
		super().__init__(config)
		self.layers = nn.ModuleList(
			[MyLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
			)


class MyLlamaForCausalLM(LlamaForCausalLM):
	def __init__(self, config: LlamaConfig):
		super().__init__(config)
		self.model = MyLlamaModel(config)
