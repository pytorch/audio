def round_up_to_nearest_power_of_two(x: int):
    assert x > 0
    return 2 ** (x - 1).bit_length()
