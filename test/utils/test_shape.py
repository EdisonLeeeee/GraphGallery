from graphgallery.utils.shape import repeat


def test_repeat():
    assert repeat(None, length=4) == [None, None, None, None]
    assert repeat([], length=4) == []
    assert repeat((), length=4) == []
    assert repeat(4, length=4) == [4, 4, 4, 4]
    assert repeat("check", length=3) == ["check", "check", "check"]
    assert repeat([2, 3, 4], length=4) == [2, 3, 4, 4]
    assert repeat([1, 2, 3, 4], length=4) == [1, 2, 3, 4]
    assert repeat([1, 2, 3, 4, 5], length=4) == [1, 2, 3, 4]
