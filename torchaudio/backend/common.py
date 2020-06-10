from typing import Any, Optional, Tuple


class SignalInfo:
    def __init__(self,
                 channels: Optional[int] = None,
                 rate: Optional[float] = None,
                 precision: Optional[int] = None,
                 length: Optional[int] = None) -> None:
        self.channels = channels
        self.rate = rate
        self.precision = precision
        self.length = length


class EncodingInfo:
    def __init__(self,
                 encoding: Any = None,
                 bits_per_sample: Optional[int] = None,
                 compression: Optional[float] = None,
                 reverse_bytes: Any = None,
                 reverse_nibbles: Any = None,
                 reverse_bits: Any = None,
                 opposite_endian: Optional[bool] = None) -> None:
        self.encoding = encoding
        self.bits_per_sample = bits_per_sample
        self.compression = compression
        self.reverse_bytes = reverse_bytes
        self.reverse_nibbles = reverse_nibbles
        self.reverse_bits = reverse_bits
        self.opposite_endian = opposite_endian
