import heapq


class PQueue:

    def __init__(self):
        self.queue = []
    
    def __len__(self):
        return len(self.queue)

    def add(self, item):
        """
        Inserts item into heap
        item should be a tuple: (priority, content)
        """
        assert (
            isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], float|int)
        ), "Item should be a tuple of length 2, (priority: int, content: any)"

        # since heapq implements min-heap, invert priority value
        priority, content = item
        priority = - abs(priority)
        heapq.heappush(self.queue, tuple([priority, content]))

    def pop(self):
        """
        Pops item with highest priority 
        """

        _, item = heapq.heappop(self.queue)
        return item






