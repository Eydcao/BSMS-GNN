class InfiniteDataLooper:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.data_iter = iter(self.data_loader)
        self.data_iter_num = 0

    def __next__(self):
        try:
            out = next(self.data_iter)
        except StopIteration:
            print(f"reached end of data loader, restart {self.data_iter_num}")
            self.data_iter_num += 1
            self.data_iter = iter(self.data_loader)
            out = next(self.data_iter)
        return out
