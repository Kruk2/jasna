from importlib.metadata import DistributionFinder

from jasna._frozen import NormalizedDistributionNameFinder


def find(name):
    return {d.name for d in NormalizedDistributionNameFinder.find_distributions(DistributionFinder.Context(name=name))}


def test_finds_a_distribution_by_its_normalized_name():
    assert find("huggingface-hub") == {"huggingface_hub"}
    assert find("Huggingface.Hub") == {"huggingface_hub"}


def test_leaves_exact_names_and_full_listings_to_the_other_finders():
    assert find("huggingface_hub") == set()
    assert find(None) == set()
