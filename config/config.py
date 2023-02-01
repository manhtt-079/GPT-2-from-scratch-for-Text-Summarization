import configparser
from dataclasses import dataclass

@dataclass
class DatasetArgs:
    max_seq_length: int
    file_path: str
    
@dataclass
class ModelArgs:
    model_checkpoint: str
    vocab_size: int
    n_positions: int
    n_embeds: int
    n_layers: int
    n_heads: int
    n_inners: int
    activation_func: str
    residual_dropout: float
    embed_dropout: float
    attn_dropout: float
    layer_norm_epsilon: float

    def __post_init__(self):
        self.vocab_size = int(self.vocab_size)
        self.n_positions = int(self.n_positions)
        self.n_embeds = int(self.n_embeds)
        self.n_layers = int(self.n_layers)
        self.n_heads = int(self.n_heads)
        self.n_inners = int(self.n_inners)
        self.residual_dropout = float(self.residual_dropout)
        self.embed_dropout = float(self.embed_dropout)
        self.attn_dropout = float(self.attn_dropout)
        self.layer_norm_epsilon = float(self.layer_norm_epsilon)

class TrainerArgs(object):
    def __init__(self, config: configparser.ConfigParser) -> None:
        self.config = config
        
        self.accelerator = self.config['trainer']['accelerator']
        self.accumulate_grad_batches = int(self.config['trainer']['accumulate_grad_batches'])
        self.amp_backend = self.config['trainer']['amp_backend']
        self.auto_lr_find = True if self.config['trainer']['auto_lr_find'].lower()=='true' else False
        self.auto_scale_batch_size = True if self.config['trainer']['auto_scale_batch_size'].lower()=='true' else False
        self.auto_select_gpus = True if self.config['trainer']['auto_select_gpus'].lower()=='true' else False
        self.batch_size = int(self.config['trainer']['batch_size'])
        self.checkpoint = self.config['trainer']['checkpoint']
        self.default_root_dir = self.config['trainer']['default_root_dir']
        self.delta = float(self.config['trainer']['delta'])
        self.devices = int(self.config['trainer']['devices'])
        self.enable_checkpointing = True if self.config['trainer']['enable_checkpointing'].lower()=='true' else False
        self.enable_progess_bar = True if self.config['trainer']['enable_progess_bar'].lower()=='true' else False
        self.enable_model_summary = True if self.config['trainer']['enable_model_summary'].lower()=='true' else False
        self.eval_steps = int(self.config['trainer']['eval_steps'])
        self.monitor = self.config['trainer']['monitor']
        self.no_decay = self.config['trainer']['no_decay'].split(',')
        self.log_every_n_steps = int(self.config['trainer']['log_every_n_steps'])
        self.log = self.config['trainer']['log']
        self.lr = float(self.config['trainer']['lr'])
        self.losses = self.config['trainer']['losses']
        self.max_epochs = int(self.config['trainer']['max_epochs'])
        self.num_beams = int(self.config['trainer']['num_beams'])
        self.num_workers = int(self.config['trainer']['num_workers'])
        self.patience = int(self.config['trainer']['patience'])
        self.precision = int(self.config['trainer']['precision'])
        self.random_state = int(self.config['trainer']['random_state'])
        self.save_top_k = int(self.config['trainer']['save_top_k'])
        self.save_on_train_epoch_end = True if self.config['trainer']['save_on_train_epoch_end'].lower()=='true' else False
        self.warmup_ratio = float(self.config['trainer']['warmup_ratio'])        
        self.weight_decay = float(self.config['trainer']['weight_decay'])
        
    def __repr__(self) -> str:
        return str(self.__dict__)
    

class Config(object):    
    def __init__(
        self,
        config_file: str
    ) -> None:
        self.config_file = config_file
        self.config = self.read_conf(conf_file=config_file)        
    
    @staticmethod
    def read_conf(conf_file) -> configparser.ConfigParser:
        config =  configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        config.read(conf_file)
    
        return config
    
    @property
    def trainer_args(self):
        return TrainerArgs(config=self.config)
        
    @property
    def model_args(self):
        return ModelArgs(**self.config['gpt2-conf'])
    
    @property
    def dataset_args(self):
        return DatasetArgs(self.config['dataset'])
   
if __name__=='__main__':
    pass

