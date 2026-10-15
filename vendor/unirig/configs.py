"""
Inlined model/tokenizer/skeleton configurations.

These were previously loaded from YAML files. Since there's only one
set of weights (UniRig), they're hardcoded here to eliminate the
YAML/Box dependency.
"""

# ============================================================================
# Checkpoint paths (relative to experiments/ on HuggingFace)
# ============================================================================

SKELETON_CHECKPOINT = "experiments/skeleton/articulation-xl_quantization_256/model.ckpt"
SKIN_CHECKPOINT = "experiments/skin/articulation-xl/model.ckpt"

# ============================================================================
# AR model config (was: unirig_ar_350m_1024_81920_float32.yaml)
# ============================================================================

AR_MODEL_CONFIG = {
    "__target__": "unirig_ar",
    "llm": {
        "pretrained_model_name_or_path": "facebook/opt-350m",
        "n_positions": 3076,
        "max_position_embeddings": 3076,
        "hidden_size": 1024,
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "ffn_dim": 4096,
        "word_embed_proj_dim": 1024,
        "do_layer_norm_before": True,
        "_attn_implementation": "flash_attention_2",
    },
    "mesh_encoder": {
        "__target__": "michelangelo_encoder",
        "pretrained_path": None,
        "freeze_encoder": False,
        "num_latents": 512,
        "embed_dim": 64,
        "point_feats": 3,
        "num_freqs": 8,
        "include_pi": False,
        "heads": 8,
        "width": 512,
        "num_encoder_layers": 16,
        "use_ln_post": True,
        "init_scale": 0.25,
        "qkv_bias": False,
        "use_checkpoint": False,
        "flash": True,
        "supervision_type": "sdf",
        "query_method": False,
        "token_num": 1024,
    },
}

# ============================================================================
# Skin model config (was: unirig_skin.yaml)
# Note: dtype/device removed — injected at runtime by direct.py
# ============================================================================

SKIN_MODEL_CONFIG = {
    "__target__": "unirig_skin",
    "num_train_vertex": 512,
    "num_heads": 16,
    "feat_dim": 768,
    "grid_size": 0.005,
    "mlp_dim": 512,
    "num_bone_attn": 8,
    "num_mesh_bone_attn": 16,
    "bone_embed_dim": 1024,
    "voxel_mask": 3.0,
    "mesh_encoder": {
        "__target__": "ptv3obj",
        "pretrained_path": None,
        "freeze_encoder": False,
        "in_channels": 9,
        "cls_mode": False,
        "shuffle_orders": True,
        "drop_path": 0.0,
        "upcast_attention": False,
        "upcast_softmax": False,
        "enc_depths": [3, 3, 3, 6, 16],
        "enc_channels": [32, 64, 128, 256, 384],
        "enc_num_head": [2, 4, 8, 16, 24],
        "enable_qknorm": True,
        "layer_norm": False,
        "res_linear": True,
    },
    "global_encoder": {
        "__target__": "michelangelo_encoder",
        "pretrained_path": None,
        "freeze_encoder": False,
        "num_latents": 512,
        "embed_dim": 64,
        "point_feats": 3,
        "num_freqs": 8,
        "include_pi": False,
        "heads": 8,
        "width": 512,
        "num_encoder_layers": 16,
        "use_ln_post": True,
        "init_scale": 0.25,
        "qkv_bias": False,
        "use_checkpoint": False,
        "flash": True,
        "supervision_type": "sdf",
        "query_method": False,
        "token_num": 1024,
    },
}

# ============================================================================
# Tokenizer config (was: tokenizer_parts_articulationxl_256.yaml)
# ============================================================================

TOKENIZER_CONFIG = {
    "method": "tokenizer_part",
    "num_discrete": 256,
    "continuous_range": [-1, 1],
    "cls_token_id": {
        "vroid": 0,
        "mixamo": 1,
        "articulationxl": 2,
    },
    "parts_token_id": {
        "body": 0,
        "hand": 1,
    },
    "order_config": {
        # Keys select which SKELETONS entries to use (values unused)
        "skeleton_path": {
            "vroid": "",
            "mixamo": "",
        },
    },
}

# ============================================================================
# Skeleton bone lists (was: configs/skeleton/mixamo.yaml, vroid.yaml)
# ============================================================================

SKELETON_MIXAMO = {
    "parts_order": ["body", "hand"],
    "parts": {
        "body": [
            "mixamorig:Hips", "mixamorig:Spine", "mixamorig:Spine1", "mixamorig:Spine2",
            "mixamorig:Neck", "mixamorig:Head",
            "mixamorig:LeftShoulder", "mixamorig:LeftArm", "mixamorig:LeftForeArm", "mixamorig:LeftHand",
            "mixamorig:RightShoulder", "mixamorig:RightArm", "mixamorig:RightForeArm", "mixamorig:RightHand",
            "mixamorig:LeftUpLeg", "mixamorig:LeftLeg", "mixamorig:LeftFoot", "mixamorig:LeftToeBase",
            "mixamorig:RightUpLeg", "mixamorig:RightLeg", "mixamorig:RightFoot", "mixamorig:RightToeBase",
        ],
        "hand": [
            "mixamorig:LeftHandThumb1", "mixamorig:LeftHandThumb2", "mixamorig:LeftHandThumb3",
            "mixamorig:LeftHandIndex1", "mixamorig:LeftHandIndex2", "mixamorig:LeftHandIndex3",
            "mixamorig:LeftHandMiddle1", "mixamorig:LeftHandMiddle2", "mixamorig:LeftHandMiddle3",
            "mixamorig:LeftHandRing1", "mixamorig:LeftHandRing2", "mixamorig:LeftHandRing3",
            "mixamorig:LeftHandPinky1", "mixamorig:LeftHandPinky2", "mixamorig:LeftHandPinky3",
            "mixamorig:RightHandIndex1", "mixamorig:RightHandIndex2", "mixamorig:RightHandIndex3",
            "mixamorig:RightHandThumb1", "mixamorig:RightHandThumb2", "mixamorig:RightHandThumb3",
            "mixamorig:RightHandMiddle1", "mixamorig:RightHandMiddle2", "mixamorig:RightHandMiddle3",
            "mixamorig:RightHandRing1", "mixamorig:RightHandRing2", "mixamorig:RightHandRing3",
            "mixamorig:RightHandPinky1", "mixamorig:RightHandPinky2", "mixamorig:RightHandPinky3",
        ],
    },
}

SKELETON_VROID = {
    "parts_order": ["body", "hand"],
    "parts": {
        "body": [
            "J_Bip_C_Hips", "J_Bip_C_Spine", "J_Bip_C_Chest", "J_Bip_C_UpperChest",
            "J_Bip_C_Neck", "J_Bip_C_Head",
            "J_Bip_L_Shoulder", "J_Bip_L_UpperArm", "J_Bip_L_LowerArm", "J_Bip_L_Hand",
            "J_Bip_R_Shoulder", "J_Bip_R_UpperArm", "J_Bip_R_LowerArm", "J_Bip_R_Hand",
            "J_Bip_L_UpperLeg", "J_Bip_L_LowerLeg", "J_Bip_L_Foot", "J_Bip_L_ToeBase",
            "J_Bip_R_UpperLeg", "J_Bip_R_LowerLeg", "J_Bip_R_Foot", "J_Bip_R_ToeBase",
        ],
        "hand": [
            "J_Bip_L_Thumb1", "J_Bip_L_Thumb2", "J_Bip_L_Thumb3",
            "J_Bip_L_Index1", "J_Bip_L_Index2", "J_Bip_L_Index3",
            "J_Bip_L_Middle1", "J_Bip_L_Middle2", "J_Bip_L_Middle3",
            "J_Bip_L_Ring1", "J_Bip_L_Ring2", "J_Bip_L_Ring3",
            "J_Bip_L_Little1", "J_Bip_L_Little2", "J_Bip_L_Little3",
            "J_Bip_R_Index1", "J_Bip_R_Index2", "J_Bip_R_Index3",
            "J_Bip_R_Thumb1", "J_Bip_R_Thumb2", "J_Bip_R_Thumb3",
            "J_Bip_R_Middle1", "J_Bip_R_Middle2", "J_Bip_R_Middle3",
            "J_Bip_R_Ring1", "J_Bip_R_Ring2", "J_Bip_R_Ring3",
            "J_Bip_R_Little1", "J_Bip_R_Little2", "J_Bip_R_Little3",
        ],
    },
}

SKELETONS = {
    "vroid": SKELETON_VROID,
    "mixamo": SKELETON_MIXAMO,
}
