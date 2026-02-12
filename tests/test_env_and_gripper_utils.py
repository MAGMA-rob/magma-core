import pytest

torch = pytest.importorskip("torch")

from magma_core.utils.env_utils import craft_random_manu_order, is_object_inside_target
from magma_core.utils.gripper_utils import find_object_in_gripper, is_object_in_gripper


def test_is_object_inside_target_single_and_batch():
    obj = torch.tensor([0.0, 0.0, 0.0])
    target = torch.tensor([0.05, 0.05, 1.0])
    assert is_object_inside_target(obj, target, thresh=0.1, keep_tensor=False) is True

    obj_b = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    target_b = torch.tensor([[0.01, 0.01, 0.0], [0.0, 0.0, 0.0]])
    out = is_object_inside_target(obj_b, target_b, thresh=0.05)
    assert out.tolist() == [True, False]


def test_craft_random_manu_order_shape_and_charset():
    order = craft_random_manu_order(8)
    assert len(order) == 8
    assert order.isalnum()
    with pytest.raises(ValueError, match="at least 2 characters"):
        craft_random_manu_order(1)


def test_find_object_in_gripper_and_is_object_in_gripper():
    tcp = torch.tensor([0.0, 0.0, 0.0])
    objects = {
        "far": torch.tensor([1.0, 1.0, 0.0]),
        "near": torch.tensor([0.01, 0.0, 0.01]),
    }
    found = find_object_in_gripper(tcp, objects, threshold=0.03, z_dist_threshold=0.05)
    assert found == "near"
    assert is_object_in_gripper(tcp, objects["near"], threshold=0.03) is True
    assert is_object_in_gripper(tcp, objects["far"], threshold=0.03) is False
