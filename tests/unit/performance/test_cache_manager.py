import pytest

from wjp_analyser.performance import cache_manager


@pytest.fixture(autouse=True)
def reset_cache_registry():
    """Ensure cache registry is clean between tests."""
    cache_manager._CACHE_INSTANCES.clear()
    yield
    cache_manager._CACHE_INSTANCES.clear()


def test_get_cache_manager_scopes_by_directory(tmp_path):
    dir_one = tmp_path / "one"
    dir_two = tmp_path / "two"
    dir_one.mkdir()
    dir_two.mkdir()

    manager_a = cache_manager.get_cache_manager(str(dir_one))
    manager_b = cache_manager.get_cache_manager(str(dir_one))
    manager_c = cache_manager.get_cache_manager(str(dir_two))

    assert manager_a is manager_b
    assert manager_a is not manager_c
    assert manager_a.cache_dir == dir_one.resolve()
    assert manager_c.cache_dir == dir_two.resolve()
