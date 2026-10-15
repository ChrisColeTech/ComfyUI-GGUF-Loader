from dataclasses import dataclass

from .michelangelo_encoder import get_encoder as get_encoder_michelangelo
from .michelangelo_encoder import AlignedShapeLatentPerceiver
from .michelangelo_encoder import get_encoder_simplified as get_encoder_michelangelo_encoder
from .michelangelo_encoder import ShapeAsLatentPerceiverEncoder
from .ptv3_encoder import get_encoder as get_encoder_ptv3obj
from .ptv3_encoder import PointTransformerV3Object

@dataclass(frozen=True)
class _MAP_MESH_ENCODER:
    ptv3obj = PointTransformerV3Object
    michelangelo = AlignedShapeLatentPerceiver
    michelangelo_encoder = ShapeAsLatentPerceiverEncoder

MAP_MESH_ENCODER = _MAP_MESH_ENCODER()


def get_mesh_encoder(**kwargs):
    MAP = {
        'ptv3obj': get_encoder_ptv3obj,
        'michelangelo': get_encoder_michelangelo,
        'michelangelo_encoder': get_encoder_michelangelo_encoder,
    }
    __target__ = kwargs['__target__']
    del kwargs['__target__']
    assert __target__ in MAP, f"expect: [{','.join(MAP.keys())}], found: {__target__}"
    return MAP[__target__](**kwargs)
