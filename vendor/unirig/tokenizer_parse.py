from .tokenizer_spec import TokenizerConfig, TokenizerSpec
from .tokenizer_part import TokenizerPart

def get_tokenizer(config: TokenizerConfig) -> TokenizerSpec:
    MAP = {
        'tokenizer_part': TokenizerPart,
    }
    assert config.method in MAP, f"expect: [{','.join(MAP.keys())}], found: {config.method}"
    return MAP[config.method](config=config)