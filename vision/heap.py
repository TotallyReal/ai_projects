import copy

class MaxHeap:

    _MIN = float('-inf')

    def __init__(self, key_to_value):

        self.heap = list(key_to_value.keys())

        self.key_to_value = copy.copy(key_to_value)
        self.key_to_index = {key: index for index, key in enumerate(self.heap)}
        self.size = len(self.key_to_value)
        depth = self.size.bit_length()

        for index in range(2 ** (depth-1) - 2,-1,-1):
            self._sift_down(index)


    def _value_at(self, index: int):
        if 0<=index<self.size:
            return self.key_to_value[self.heap[index]]
        return MaxHeap._MIN

    def _swap(self, idx1: int, idx2):
        key1, key2 = self.heap[idx1], self.heap[idx2]
        self.heap[idx1], self.heap[idx2] = key2, key1
        self.key_to_index[key1] = idx2
        self.key_to_index[key2] = idx1

    def _sift_down(self, index: int):
        child1_idx = 2*index + 1
        child2_idx = 2*index + 2
        while child1_idx < self.size:
            value1 = self._value_at(child1_idx)
            value2 = self._value_at(child2_idx)
            big_child_idx, big_value = (child1_idx, value1) if (value1>=value2) else (child2_idx, value2)

            if self._value_at(index) >= big_value:
                break

            self._swap(index, big_child_idx)
            index = big_child_idx
            child1_idx = 2*index + 1
            child2_idx = 2*index + 2


    def _validate_if_need(self, parent_idx: int, child_idx: int):
        if self.key_to_value[self.heap[parent_idx - 1]] < self.key_to_value[self.heap[child_idx - 1]]:
            parent_key = self.heap[parent_idx - 1]
            child_key = self.heap[child_idx - 1]
            self.heap[parent_idx - 1] = child_key
            self.key_to_index[child_key] = parent_idx

            self.heap[child_idx - 1]  = parent_key
            self.key_to_index[parent_key] = child_idx
            return False
        return True

    def try_update_value(self, key, plus_value):
        index = self.key_to_index.get(key, -1)
        if index == -1 or self.size<= index:
            return False
        self.update_value(key, plus_value)
        return True

    def update_value(self, key, plus_value):
        self.key_to_value[key] += plus_value
        index = self.key_to_index[key]
        parent_index = (index-1) // 2
        while parent_index >= 0:
            if self._value_at(parent_index) >= self._value_at(index):
                break

            self._swap(parent_index, index)
            index = parent_index
            parent_index = (index-1) // 2

    def pop(self):
        key = self.heap[0]
        value = self.key_to_value[key]

        self.size -= 1
        self._swap(0, self.size)
        self._sift_down(0)

        return key, value

    def empty(self):
        return self.size == 0

    def __len__(self):
        return self.size