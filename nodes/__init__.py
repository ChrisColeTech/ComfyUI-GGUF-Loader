from .gguf import (GGUFModelPatcher, UnetLoaderGGUF, UnetLoaderGGUFAdvanced,
                    CLIPLoaderGGUF, DualCLIPLoaderGGUF, TripleCLIPLoaderGGUF,
                    QuadrupleCLIPLoaderGGUF, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS)

# nodes/preprocessors.py (DepthMap, Canny) does NOT register its own nodes -
# it's a shared implementation krea2/qwen_image/flux_klein import from
# directly, registered only under those pipelines' own historical node
# names (Krea2DepthMap, Flux2KleinDepthMap, QwenImageCanny). See that
# module's docstring for why - avoids colliding with the standalone
# ComfyUI-ControlNet-Nodes package's own "DepthMap"/"Canny" registrations.

# Imported last: nodes_extra subclasses CLIPLoaderGGUF, so it has to come after
# the class definitions above rather than at the top of the file.
from .extra import (NODE_CLASS_MAPPINGS as _EXTRA_CLASSES,
                     NODE_DISPLAY_NAME_MAPPINGS as _EXTRA_NAMES)

NODE_CLASS_MAPPINGS.update(_EXTRA_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS.update(_EXTRA_NAMES)

# Imported last because the Scenema nodes reuse GGUF loader helpers above.
from .scenema import (NODE_CLASS_MAPPINGS as _SCENEMA_CLASSES,
                       NODE_DISPLAY_NAME_MAPPINGS as _SCENEMA_NAMES)

NODE_CLASS_MAPPINGS.update(_SCENEMA_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS.update(_SCENEMA_NAMES)

from .minimax_music import (NODE_CLASS_MAPPINGS as _MUSIC_CLASSES,
                             NODE_DISPLAY_NAME_MAPPINGS as _MUSIC_NAMES)

NODE_CLASS_MAPPINGS.update(_MUSIC_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS.update(_MUSIC_NAMES)

from .stems import (NODE_CLASS_MAPPINGS as _STEM_CLASSES,
                     NODE_DISPLAY_NAME_MAPPINGS as _STEM_NAMES)

NODE_CLASS_MAPPINGS.update(_STEM_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS.update(_STEM_NAMES)

from .minimax_h3 import (NODE_CLASS_MAPPINGS as _H3_CLASSES,
                          NODE_DISPLAY_NAME_MAPPINGS as _H3_NAMES)

NODE_CLASS_MAPPINGS.update(_H3_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS.update(_H3_NAMES)


from .ltx25 import (NODE_CLASS_MAPPINGS as _LTX25_CLASSES,
                     NODE_DISPLAY_NAME_MAPPINGS as _LTX25_NAMES)

NODE_CLASS_MAPPINGS.update(_LTX25_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS.update(_LTX25_NAMES)

from .zimage import (NODE_CLASS_MAPPINGS as _ZIMAGE_CLASSES,
                      NODE_DISPLAY_NAME_MAPPINGS as _ZIMAGE_NAMES)

NODE_CLASS_MAPPINGS.update(_ZIMAGE_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS.update(_ZIMAGE_NAMES)

from .ltx23 import (NODE_CLASS_MAPPINGS as _LTX23_CLASSES,
                     NODE_DISPLAY_NAME_MAPPINGS as _LTX23_NAMES)

NODE_CLASS_MAPPINGS.update(_LTX23_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS.update(_LTX23_NAMES)

from .lmstudio import (NODE_CLASS_MAPPINGS as _LMSTUDIO_CLASSES,
                        NODE_DISPLAY_NAME_MAPPINGS as _LMSTUDIO_NAMES)

NODE_CLASS_MAPPINGS.update(_LMSTUDIO_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS.update(_LMSTUDIO_NAMES)

from .qwen_tts import (NODE_CLASS_MAPPINGS as _QWEN_TTS_CLASSES,
                        NODE_DISPLAY_NAME_MAPPINGS as _QWEN_TTS_NAMES)

NODE_CLASS_MAPPINGS.update(_QWEN_TTS_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS.update(_QWEN_TTS_NAMES)

from .krea2 import (NODE_CLASS_MAPPINGS as _KREA2_CLASSES,
                     NODE_DISPLAY_NAME_MAPPINGS as _KREA2_NAMES)

NODE_CLASS_MAPPINGS.update(_KREA2_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS.update(_KREA2_NAMES)

from .qwen_image import (NODE_CLASS_MAPPINGS as _QWEN_IMAGE_CLASSES,
                          NODE_DISPLAY_NAME_MAPPINGS as _QWEN_IMAGE_NAMES)

NODE_CLASS_MAPPINGS.update(_QWEN_IMAGE_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS.update(_QWEN_IMAGE_NAMES)

from .flux_klein import (NODE_CLASS_MAPPINGS as _FLUX_KLEIN_CLASSES,
                          NODE_DISPLAY_NAME_MAPPINGS as _FLUX_KLEIN_NAMES)

NODE_CLASS_MAPPINGS.update(_FLUX_KLEIN_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS.update(_FLUX_KLEIN_NAMES)
