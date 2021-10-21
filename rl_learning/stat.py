class StreamignMean:
    def __init__(self):
        self._partial_sum = 0
        self._count = 0

    def add_value(self, new_value: float) -> None:
        self._partial_sum += new_value
        self._count += 1

    def mean(self) -> float:
        return self._partial_sum / self._count

    def total_meas(self) -> int:
        return self._count
