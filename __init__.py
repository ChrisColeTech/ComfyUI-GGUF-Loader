# only import if running as a custom node
try:
    import comfy.utils
except ImportError:
    pass
else:
    import random
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    tech_rambling = [
        "Zap zap zoom!", "Sproing-a-ling!", "Flux capacitor charged!",
        "Circuit party started!", "Electrons dancing!", "Voltage va-va-voom!",
        "Capacitor doing the cha-cha!", "Resistor raving!", "GGUF go brrr!",
        "Quantized and caffeinated!",
    ]

    print(
        f"\033[1;34m[CCTech Suite]: 🤖🤖🤖 \033[96m\033[3m"
        f"{random.choice(tech_rambling)}\033[0m 🤖🤖🤖"
    )
    print(
        f"\033[1;34m[CCTech Suite]:\033[0m Activated "
        f"\033[96m{len(NODE_CLASS_MAPPINGS)}\033[0m GGUF loader nodes "
        f"(🤖 CCTech/GGUF)."
    )

    WEB_DIRECTORY = "./web"
    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
