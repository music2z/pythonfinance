class JinyTrading:

    def __init__(self):
        print('JinyTrading init')
        self.datas = None
        self.strategy = None
        self.broker = self.Broker(self)
        self.stake = None

    def adddata(self, datas):
        self.datas = datas

    def addstrategy(self, strategy):
        self.strategy = strategy

    def run(self):
        print('run!!!')

    def addsizer(self, stake):
        self.stake = stake

    def plot(self):
        print('plot')

    class Broker:
        def __init__(self, outer):
            print('Broker init')
            self.outer = outer
            self.cash = 1000.0
            self.commision = 0.0

        def setcash(self, cash):
            self.cash = cash

        def setcommission(self, commision):
            self.commision = commision


class Strategy:

    params = (
        ('maperiod', 15),
    )

    def log(self, txt, dt=None):
        print('%s %s' % (dt.isoformat(), txt))

    def __init__(self):
        self.dataclose = None
        self.order = None
        self.buyprice = None
        self.buycomm = None

    def notify_order(self, order):
        pass

    def notify_trade(self, trade):
        pass

    def next(self):
        pass



