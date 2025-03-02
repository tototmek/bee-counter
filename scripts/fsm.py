class DetectorFsm:
    def __init__(self):
        self.state = 0
        self.output = 0
        self.timer = 0
        self.timeout = 1000

    def step(self, input: float) -> float:
        if self.state == 0:
            if input == 1:
                self.state = 1
                self.timer = 0
            elif input == -1:
                self.state = 2
                self.timer = 0

        elif self.state == 1:
            self.timer += 1
            if input == -1:
                self.state = 0
                self.output += 1
            elif self.timer > self.timeout:
                self.state = 0

        elif self.state == 2:
            self.timer += 1
            if input == 1:
                self.state = 0
                self.output -= 1
            elif self.timer > self.timeout:
                self.state = 0

        return self.output
