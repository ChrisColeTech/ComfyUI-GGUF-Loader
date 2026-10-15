from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict

import numpy as np
from numpy import ndarray

from typing import Union, List, Tuple
from dataclasses import dataclass

from .data_order import OrderConfig, Order, get_order

@dataclass(frozen=True)
class TokenizerConfig():
    # which tokenizer to use
    method: str

    # coord discrete
    num_discrete: int

    # normalization range
    continuous_range: Tuple[float, float]

    # cls token id
    cls_token_id: Dict[str, int]

    # parts token id
    parts_token_id: Dict[str, int]

    order_config: Union[OrderConfig, None]

    @staticmethod
    def parse(config, base_path=None) -> 'TokenizerConfig':
        order_config = config.get('order_config', None)

        return TokenizerConfig(
            method=config['method'],
            num_discrete=config['num_discrete'],
            continuous_range=config['continuous_range'],
            cls_token_id=config['cls_token_id'],
            parts_token_id=config.get('parts_token_id', {}),
            order_config=OrderConfig.parse(order_config, base_path=base_path) if order_config is not None else None,
        )

@dataclass(frozen=True)
class DetokenizeOutput():
    # original tokens
    tokens: ndarray

    # (J, 6), (parent position, position)
    bones: ndarray

    # (J), parent of each bone
    parents: List[Union[int, None]]

    # (J, 3), tails of bones(this is an attribute to indicate direction, not bones[i, 3:6])
    tails: Union[ndarray, None]

    # (B, J), bool, whether the bone has skin
    no_skin: Union[ndarray, None]

    # string of class in tokenizer
    cls: Union[str, None]

    # part names in order
    parts: List[str]

    # names of joints
    names: Union[None, List[str]]

    # normalization cube
    continuous_range: Tuple[float, float]

    @property
    def joints(self):
        return self.bones[:, 3:]

class TokenizerSpec(ABC):
    """
    Abstract class for tokenizer
    """

    def __init__(self, **kwargs):
        super().__init__()
        pass

    def detokenize(self, ids: ndarray, **kwargs) -> DetokenizeOutput:
        raise NotImplementedError("{} has no method 'detokenize'".format(type(self).__name__))

    @abstractmethod
    def cls_name_to_token(self, cls: str) -> int:
        """Cls name to token"""
        pass

    @property
    @abstractmethod
    def vocab_size(self):
        """The vocabulary size"""
        pass

    @property
    def pad(self):
        raise NotImplementedError("{} has no attribute 'pad'".format(type(self).__name__))

    @property
    def bos(self):
        raise NotImplementedError("{} has no attribute 'bos'".format(type(self).__name__))

    @property
    def eos(self):
        raise NotImplementedError("{} has no attribute 'eos'".format(type(self).__name__))

    @property
    def num_discrete(self):
        raise NotImplementedError("{} has no attribute 'num_discrete'".format(type(self).__name__))

    @property
    @abstractmethod
    def continuous_range(self) -> Tuple[float, float]:
        pass

    def next_posible_token(self, ids: ndarray) -> List[int]:
        pass

def make_skeleton(
    joints: ndarray,
    p_joints: ndarray,
    tails_dict: Dict[int, ndarray],
    convert_leaf_bones_to_tails: bool,
    extrude_tail_for_leaf: bool,
    extrude_tail_for_branch: bool,
    extrude_scale: float=0.5,
    strict: bool=False,
) -> Tuple[ndarray, ndarray, List[int], List[Union[None, int]]]:
    '''
    Args:
        joints: heads of bones

        p_joints: parent position of joints

        tails_dict: tail position of the i-th joint

        convert_leaf_bones_to_tails: remove leaf bones and make them tails of their parents

        extrude_tail_for_leaf: add a tail for leaf bone

        extrude_tail_for_branch: add a tail for joint with multiple children

        extrude_scale: length scale of tail offset

        strict: if true, raise error when there are joints in the same location

    Returns:
        bones, tails, available_bones_id, parents
    '''
    assert (convert_leaf_bones_to_tails & extrude_tail_for_leaf)==False, 'cannot extrude tail for leaf when convert_leaf_bones_to_tails is True'
    assert joints.shape[0] == p_joints.shape[0]
    # build parents
    bones = [] # (parent_position, position)
    parents = []
    for (i, joint) in enumerate(joints):
        if len(bones) == 0:
            bones.append(np.concatenate([joint, joint])) # root
            parents.append(None)
            continue
        p_joint = p_joints[i]
        dis = 999999
        pid = None
        for j in reversed(range(i)):
            n_dis = ((bones[j][3:] - p_joint)**2).sum()
            if n_dis < dis:
                pid = j
                dis = n_dis
        bones.append(np.concatenate([joints[pid], joint]))
        parents.append(pid)
    bones = np.stack(bones)

    children = defaultdict(list)
    for (i, pid) in enumerate(parents):
        if pid is None:
            continue
        children[pid].append(i)

    available_bones_id = []
    if convert_leaf_bones_to_tails:
        for (i, pid) in enumerate(parents):
            if len(children[i]) != 0:
                available_bones_id.append(i)
                continue
            tails_dict[pid] = bones[i, 3:]
    else:
        available_bones_id = [i for i in range(bones.shape[0])]

    # tail for leaf
    for (i, pid) in enumerate(parents):
        if len(children[i]) != 0:
            continue
        if extrude_tail_for_leaf:
            d = bones[i, 3:] - bones[pid, 3:]
            length = np.linalg.norm(d)
            if strict:
                assert length > 1e-9, 'two joints in the same point found'
            elif length <= 1e-9:
                d = np.array([0., 0., 1.])
            tails_dict[i] = bones[i, 3:] + d * extrude_scale
        else:
            tails_dict[i] = bones[i, 3:]

    # tail for branch
    for (i, pid) in enumerate(parents):
        if len(children[i]) <= 1:
            continue
        if extrude_tail_for_branch:
            if pid is None: # root
                av_len = 0
                for child in children[i]:
                    av_len += np.linalg.norm(bones[i, 3:] - bones[child, 3:])
                av_len /= len(children[i])
                d = bones[i, 3:] + np.array([0., 0., extrude_scale * av_len])
            else:
                d = bones[i, 3:] - bones[pid, 3:]
                length = np.linalg.norm(d)
                if strict:
                    assert length > 1e-9, 'two joints in the same point found'
                elif length <= 1e-9:
                    d = np.array([0., 0., 1.])
            tails_dict[i] = bones[i, 3:] + d * extrude_scale
        else:
            tails_dict[i] = bones[i, 3:]

    # assign new tail
    for (i, pid) in enumerate(parents):
        if len(children[i]) != 1:
            continue
        child = children[i][0]
        tails_dict[i] = bones[child, 3:]

    tails = []
    for i in range(bones.shape[0]):
        tails.append(tails_dict[i])
    tails = np.stack(tails)
    return bones, tails, available_bones_id, parents
