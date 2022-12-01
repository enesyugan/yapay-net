from transformers import PretrainedConfig
from typing import List


class Seq2SeqLstmConfig(PretrainedConfig):
   # model_type = "resnet"

    def __init__(
        self,
        vocab_size: int = 4003,
        input_channels: int = 40,
        enc_layers: int = 6,
	dec_layers: int = 2,
	attention_heads: int = 8,

        enc_dim: int = 1024,
        dec_dim: int = 1024,
	emb_dim: int = 512,
        enc_unidirectional: bool = False,
	dec_unidirectional: bool = True,
	shared_emb: bool = False,

        freq_kn: int = 3,
	freq_std: int = 2,

	use_cnn: bool = True,
        projection_dim: int = 0,

	enc_dropout: float = 0.2,
	enc_dropconnect: float = 0.,
	dec_dropout: float = 0.2,
	dec_dropconnect: float = 0.,
	emb_dropout: float = 0.,

        pack_batch: bool = True,

	bos_token_id: int = 1,
	eos_token_id: int = 2,
        init_std: float = 0.02,

        is_encoder_decoder=True,
  
        **kwargs,
    ):
       # if block_type not in ["basic", "bottleneck"]:
       #     raise ValueError(f"`block` must be 'basic' or bottleneck', got {block}.")     


        self.vocab_size = vocab_size
        self.input_channels = input_channels
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.attention_heads = attention_heads

        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        self.emb_dim = emb_dim
        self.enc_unidirectional = enc_unidirectional
        self.dec_unidirectional = dec_unidirectional
        self.shared_emb = shared_emb

        self.freq_kn = freq_kn
        self.freq_std = freq_std

        self.use_cnn = use_cnn
        self.projection_dim = projection_dim

        self.enc_dropout = enc_dropout
        self.enc_dropconnect = enc_dropconnect
        self.dec_dropout = dec_dropout
        self.dec_dropconnect = dec_dropconnect
        self.emb_dropout = emb_dropout
 
        self.pack_batch = pack_batch    

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.init_std = init_std
        #self.is_encoder_decoder = is_encoder_decoder
        super().__init__(
		is_encoder_decoder=is_encoder_decoder,
		**kwargs
		)
