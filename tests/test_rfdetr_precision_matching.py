import pytest
import torch

from scripts.probe_windows_rfdetr_precision import compare_proposals


def proposals():
    return dict(dets=torch.tensor([[[.2, .2, .2, .2], [.7, .7, .2, .2]]]),
                labels=torch.tensor([[[3., -3.], [-3., 3.]]]),
                masks=torch.tensor([[[[1., -1.], [-1., 1.]], [[-1., 1.], [1., -1.]]]]))


def test_topk_permutation_is_not_a_numerical_or_quality_failure():
    a = proposals()
    b = {name: value[:, [1, 0]].clone() for name, value in a.items()}
    result = compare_proposals(a, b)
    assert result["strict_matched_numbers"]
    assert result["batches"][0]["reordered_queries"] == 2
    assert result["batches"][0]["high_overlap_matches"] == 2
    assert result["quality_acceptance"].startswith("NOT_CERTIFIED")


def test_missing_active_region_cannot_be_hidden_by_assignment():
    a = proposals()
    b = {name: value.clone() for name, value in a.items()}
    b["dets"][0, 0, :2] = .95
    result = compare_proposals(a, b)
    assert not result["strict_matched_numbers"]
    assert result["batches"][0]["reference_active_without_high_overlap"] > 0


def test_changed_mask_reported_separately_from_boxes():
    a = proposals()
    b = {name: value.clone() for name, value in a.items()}
    b["masks"] *= -1
    result = compare_proposals(a, b)
    assert result["batches"][0]["numerical"]["dets"]["allclose"]
    assert result["batches"][0]["high_overlap_matches"] == 0


def test_nonfinite_and_changed_shapes_refuse_comparison():
    a = proposals()
    b = {name: value.clone() for name, value in a.items()}
    b["labels"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        compare_proposals(a, b)
    b = {name: value[:, :1] for name, value in a.items()}
    with pytest.raises(ValueError, match="shapes changed"):
        compare_proposals(a, b)


def test_empty_masks_and_no_active_proposals_not_claimed_as_quality_pass():
    a = proposals()
    a["labels"].fill_(-20)
    a["masks"].fill_(-20)
    result = compare_proposals(a, a)
    assert result["batches"][0]["reference_active"] == 0
    assert result["batches"][0]["high_overlap_matches"] == 0
    assert result["quality_acceptance"].startswith("NOT_CERTIFIED")
