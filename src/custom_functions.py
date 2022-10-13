import torch
import numpy as np


def rover_dynamics(u, x0):
    m = 5  # mass
    h = 0.1  # deltat
    T = 100  # number of steps
    eta = 1.0  # friction coeff

    # state, control
    dim_s = 4
    dim_c = 2

    # dynamics
    A = torch.tensor(
        [
            [1, 0, h, 0],
            [0, 1, 0, h],
            [0, 0, (1 - eta * h / m), 0],
            [0, 0, 0, (1 - eta * h / m)],
        ]
    ).to(torch.float32)
    B = torch.tensor([[0, 0], [0, 0], [h / m, 0], [0, h / m]]).to(torch.float32)

    # state control (time is a row)
    x = torch.zeros((T, dim_s)).to(torch.float32)

    # reshape the control
    u = torch.reshape(u, (T, dim_c)).to(torch.float32)

    # initial condition
    x[0] = x0

    # dynamics
    # x_{t+1}  = Ax_t + Bu_t for t=0,...,T-1
    for t in range(0, T - 1):
        x[t + 1] = A @ x[t] + B @ u[t]
    return x


def rover_obj_torch(u):
    """
    The rover problem:
    The goal is to learn a controller to drive a rover through four
    waypoints.
    state: 4dim position, velocity
    control: 2dim x,y forces

    input:
    u: length 2T array, open-loop controller
    return:
    cost: float, cost associated with the controller
    """
    # assert len(u) == 200
    # initial condition
    x0 = torch.tensor([5.0, 20.0, 0.0, 0.0]).to(torch.float32)
    # compute dynamics
    x = rover_dynamics(u, x0)
    # waypoints
    W = torch.tensor([[8, 15, 3, -4], [16, 7, 6, -4], [16, 12, -6, -4], [0, 0, 0, 0]])
    way_times = torch.tensor([10, 40, 70, 100]) - 1  # .astype(int)
    # way_times = (torch.tensor([.1,.4,.7,1]) * (0.5 * len(u)) - 1).long()
    q1 = 1e0  # penalty on missing waypoint
    q2 = 1e-4  # penalty on control
    # compute cost
    cost = q1 * torch.sum((x[way_times] - W) ** 2) + q2 * torch.sum(u ** 2)

    return cost


def batch_rover(u, noise_std=0.0):
    batch, dim = u.size()

    batch_func = u.new_zeros((batch,))
    batch_grad = u.new_zeros(u.size())

    for i in range(batch):
        with torch.enable_grad():
            x = u[i].cpu().clone().detach().requires_grad_(True)
            func = rover_obj_torch(x)
            func.backward()

        with torch.no_grad():
            batch_func[i] = func
            batch_grad[i] = x.grad

    return batch_func.to(u.device), batch_grad.to(u.device)


class Rover:
    def __init__(self, noise_std=0.0):
        self.noise_std = noise_std
        self.lb = -3 * np.ones(200)
        self.ub = 3 * np.ones(200)

    def __call__(self, x):
        return batch_rover(x, noise_std=self.noise_std)


def rover(x):
    if len(x.shape) < 2:
        x = x.unsqueeze(0)

    return -batch_rover(x)[0] / 1000


if __name__ == "__main__":
    dim = 200
    x = torch.ones(
        dim,
    )
    x.requires_grad_(True)
    fx = rover_obj_torch(x)
    print("Fx", fx)
    df_dx = torch.autograd.grad(fx, x)[0]
    print("grad,", df_dx[0:10])
    print("grad shape", df_dx.shape)
    print("norm", torch.linalg.norm(df_dx))

    x = x - 0.8 * df_dx
    x.requires_grad_(True)
    fx = rover_obj_torch(x)
    print("Fx", fx)
    df_dx = torch.autograd.grad(fx, x)[0]
    print("grad,", df_dx[0:10])
    print("grad shape", df_dx.shape)
    print("norm", torch.linalg.norm(df_dx))
