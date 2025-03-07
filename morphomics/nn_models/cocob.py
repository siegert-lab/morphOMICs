import torch


class COCOBBackprop(torch.optim.Optimizer):
    def __init__(self, params, alpha=100.0, eps=1e-8):
        self.alpha = alpha
        self.eps = eps
        defaults = dict(alpha=alpha, eps=eps)
        super(COCOBBackprop, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # optimize for each parameter groups
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = -param.grad.data
                state = self.state[param]

                if len(state) == 0:
                    state = self.initial_state(state, param)

                initial_weight = state["initial_weight"]
                reward = state["reward"]
                bet = state["bet"]
                negative_grads_sum = state["negative_grads_sum"]
                absolute_grads_sum = state["absolute_grads_sum"]
                max_observed_scale = state["max_observed_scale"]

                # update parameters
                max_observed_scale = torch.max(max_observed_scale, torch.abs(grad))
                absolute_grads_sum += torch.abs(grad)
                negative_grads_sum += grad

                win_amount = bet * grad
                reward = torch.max(reward+win_amount, torch.zeros_like(reward))
                bet_fraction = negative_grads_sum / (max_observed_scale * (torch.max(absolute_grads_sum + max_observed_scale, self.alpha * max_observed_scale)))
                bet = bet_fraction * (max_observed_scale + reward)

                # set new state
                param.data = initial_weight + bet
                state["negative_grads_sum"] = negative_grads_sum
                state["absolute_grads_sum"] = absolute_grads_sum
                state["max_observed_scale"] = max_observed_scale
                state["reward"] = reward
                state["bet"] = bet
                state["bet_fraction"] = bet_fraction

        return loss

    def initial_state(self, state, param):
        assert len(state) == 0

        state["initial_weight"] = param.data
        state["reward"] = param.new_zeros(param.shape)
        state["bet"] = param.new_zeros(param.shape)
        state["negative_grads_sum"] = param.new_zeros(param.shape)
        state["absolute_grads_sum"] = param.new_zeros(param.shape)
        state["max_observed_scale"] = self.eps * param.new_ones(param.shape)

        return state