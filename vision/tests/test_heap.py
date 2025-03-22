from vision.heap import MaxHeap
import random
import pytest

@pytest.fixture(autouse=True)
def set_random_seed():
    seed = random.randint(0, 1000000000)
    print(f'Seed is {seed}')
    random.seed(seed)

def test_heap_creation():
    max_value = 100
    values = {i: random.randint(0, max_value) for i in range(50)}
    max_heap = MaxHeap(values)
    last_value = max_value
    while not max_heap.empty():
        key, value = max_heap.pop()
        assert value <= last_value
        last_value = value