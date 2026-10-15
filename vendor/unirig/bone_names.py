"""Bone-name conventions and remapping for UniRig output.

UniRig's detokenizer names bones from a template when the generation class is
recognized. The released (articulation-xl) checkpoint still accepts the
``vroid`` class token to emit the 52-bone humanoid structure (body + hands),
yielding ``J_Bip_*`` names. We then remap those to the de-facto Mixamo
``mixamorig:*`` convention so the rig is directly compatible with Mixamo
animations and game-engine humanoid rigs (and HY-Motion's retarget path).

Maps mirror ComfyUI-UniRig's ``skeleton_extraction.py``.
"""

from __future__ import annotations

from typing import List

# Generation class token per requested template. The model emits the vroid
# 52-bone humanoid for both vroid and mixamo templates.
TEMPLATE_TO_CLS = {
    "mixamo": "vroid",
    "vroid": "vroid",
    "articulationxl": "articulationxl",
    "smpl": "vroid",
}

VROID_TO_MIXAMO_BONE_MAP = {
    # Body (22)
    "J_Bip_C_Hips": "mixamorig:Hips",
    "J_Bip_C_Spine": "mixamorig:Spine",
    "J_Bip_C_Chest": "mixamorig:Spine1",
    "J_Bip_C_UpperChest": "mixamorig:Spine2",
    "J_Bip_C_Neck": "mixamorig:Neck",
    "J_Bip_C_Head": "mixamorig:Head",
    "J_Bip_L_Shoulder": "mixamorig:LeftShoulder",
    "J_Bip_L_UpperArm": "mixamorig:LeftArm",
    "J_Bip_L_LowerArm": "mixamorig:LeftForeArm",
    "J_Bip_L_Hand": "mixamorig:LeftHand",
    "J_Bip_R_Shoulder": "mixamorig:RightShoulder",
    "J_Bip_R_UpperArm": "mixamorig:RightArm",
    "J_Bip_R_LowerArm": "mixamorig:RightForeArm",
    "J_Bip_R_Hand": "mixamorig:RightHand",
    "J_Bip_L_UpperLeg": "mixamorig:LeftUpLeg",
    "J_Bip_L_LowerLeg": "mixamorig:LeftLeg",
    "J_Bip_L_Foot": "mixamorig:LeftFoot",
    "J_Bip_L_ToeBase": "mixamorig:LeftToeBase",
    "J_Bip_R_UpperLeg": "mixamorig:RightUpLeg",
    "J_Bip_R_LowerLeg": "mixamorig:RightLeg",
    "J_Bip_R_Foot": "mixamorig:RightFoot",
    "J_Bip_R_ToeBase": "mixamorig:RightToeBase",
    # Left hand (15)
    "J_Bip_L_Thumb1": "mixamorig:LeftHandThumb1",
    "J_Bip_L_Thumb2": "mixamorig:LeftHandThumb2",
    "J_Bip_L_Thumb3": "mixamorig:LeftHandThumb3",
    "J_Bip_L_Index1": "mixamorig:LeftHandIndex1",
    "J_Bip_L_Index2": "mixamorig:LeftHandIndex2",
    "J_Bip_L_Index3": "mixamorig:LeftHandIndex3",
    "J_Bip_L_Middle1": "mixamorig:LeftHandMiddle1",
    "J_Bip_L_Middle2": "mixamorig:LeftHandMiddle2",
    "J_Bip_L_Middle3": "mixamorig:LeftHandMiddle3",
    "J_Bip_L_Ring1": "mixamorig:LeftHandRing1",
    "J_Bip_L_Ring2": "mixamorig:LeftHandRing2",
    "J_Bip_L_Ring3": "mixamorig:LeftHandRing3",
    "J_Bip_L_Little1": "mixamorig:LeftHandPinky1",
    "J_Bip_L_Little2": "mixamorig:LeftHandPinky2",
    "J_Bip_L_Little3": "mixamorig:LeftHandPinky3",
    # Right hand (15)
    "J_Bip_R_Thumb1": "mixamorig:RightHandThumb1",
    "J_Bip_R_Thumb2": "mixamorig:RightHandThumb2",
    "J_Bip_R_Thumb3": "mixamorig:RightHandThumb3",
    "J_Bip_R_Index1": "mixamorig:RightHandIndex1",
    "J_Bip_R_Index2": "mixamorig:RightHandIndex2",
    "J_Bip_R_Index3": "mixamorig:RightHandIndex3",
    "J_Bip_R_Middle1": "mixamorig:RightHandMiddle1",
    "J_Bip_R_Middle2": "mixamorig:RightHandMiddle2",
    "J_Bip_R_Middle3": "mixamorig:RightHandMiddle3",
    "J_Bip_R_Ring1": "mixamorig:RightHandRing1",
    "J_Bip_R_Ring2": "mixamorig:RightHandRing2",
    "J_Bip_R_Ring3": "mixamorig:RightHandRing3",
    "J_Bip_R_Little1": "mixamorig:RightHandPinky1",
    "J_Bip_R_Little2": "mixamorig:RightHandPinky2",
    "J_Bip_R_Little3": "mixamorig:RightHandPinky3",
}


def cls_for_template(template: str) -> str:
    """Return the generation class token for a requested name template."""
    return TEMPLATE_TO_CLS.get(template, "articulationxl")


def remap_names(names: List[str], template: str) -> List[str]:
    """Remap bone names to the requested convention (no-op if not mixamo)."""
    if template != "mixamo" or names is None:
        return names
    out = []
    n = 0
    for nm in names:
        if nm in VROID_TO_MIXAMO_BONE_MAP:
            out.append(VROID_TO_MIXAMO_BONE_MAP[nm])
            n += 1
        else:
            out.append(nm)
    return out
