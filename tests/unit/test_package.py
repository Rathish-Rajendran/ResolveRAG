import resolverag


def test_package_is_importable() -> None:
    assert resolverag.__name__ == "resolverag"
