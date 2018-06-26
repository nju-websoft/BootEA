class Params:
    def __init__(self):
        self.embed_size = 75
        self.batch_size = 20000
        self.epochs = 500
        self.learning_rate = 0.01

        self.top_k = 20

        self.ent_top_k = [1, 5, 10, 50]
        self.nums_threads = 10

        self.lambda_1 = 0.01
        self.lambda_2 = 2.0
        self.lambda_3 = 0.7
        self.mu_1 = 0.2

        self.epsilon = 0.9
        self.nums_neg = 10

        self.heuristic = True

    def print(self):
        print("Parameters used in this running are as follows:")
        for item in self.__dict__.items():
            print("%s: %s" % item)
        print()


P = Params()
P.print()
