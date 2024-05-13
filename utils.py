import heapq
import numpy as np



class PQueue:

    def __init__(self):
        self._queue = []

        # {content: priority}
        self._entries = {}

    def __len__(self):
        return len(self._queue)

    def add(self, item):
        """
        Inserts item into heap
        item should be a tuple: (priority, content)
        """
        assert (
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], float | int)
        ), "Item should be a tuple of length 2, (priority: int, content: any)"

        # since heapq implements min-heap, invert priority value
        priority, content = item
        priority = -abs(priority)
        if content in self._entries:
            if priority < self._entries[content]:

                # TODO: implement faster search indexing of queue
                item = tuple([priority, content])
                arr = np.array(self._queue, dtype=object)[:, 1]
                arr = np.array([hash(i) for i in arr])
                idx = int(np.squeeze(np.where(arr == hash(content))))
                self._queue[idx] = item
                self._entries[content] = priority
        else:
            # insert item in heap and hashmap
            heapq.heappush(self._queue, tuple([priority, content]))
            self._entries[content] = priority

    def pop(self):
        """
        Pops item with highest priority
        """
        if len(self._queue):
            _, item = heapq.heappop(self._queue)
            self._entries.pop(item)
            item = list(item)
            for idx, i in enumerate(item):
                if isinstance(i, tuple):
                    item[idx] = list(i)
            return item
