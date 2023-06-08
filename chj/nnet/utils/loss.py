import copy
class Loss_Report:
    def __init__(self):
        self.losses=[]
        self.move_losses = []
        self.is_net = False
        pass

    def reset(self):
        self.losses = []

    def get_move_loss(self, i):
        if len(self.losses) > i: return self.move_losses[i]
        return None

    def update_loss(self, loss):
        if len(self.losses)==0 or len(loss)!= len(self.move_losses):
            self.move_losses = copy.copy(list(loss))
        self.losses = list(loss)

        for i in range( len(self.losses) ):
            if self.is_net or self.losses[i] <=0: continue
            self.move_losses[i] = 0.99*self.move_losses[i]+0.01*self.losses[i]

    def report(self):
        str=""
        _n = len(self.losses)
        for i in range( _n ):
            str+="%d/%d loss %.6f, move_loss %.6f\n" %(i+1, _n, self.losses[i], self.move_losses[i])
        return str

