import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import casadi
from orca_utils import projectOnVO
eps = 1e-6
class Agent:
    
    def __init__(self, A, B, G, g, H, h, radius, _id, x_0, Q, R, x_F, tau=1):
        self.A = A
        self.B = B
        self.G = G
        self.g = g
        self.H = H
        self.h = h
        self.radius = radius
        self._id = _id
        self.x = [x_0]
        self.Q = Q
        self.R = R
        self.x_F = x_F
        self.u = []
        self.tau = tau
        self.color_list = ["b", "g", "r", "c", "m", "y", "k", "w"]


    def find_norm(self):
        return np.linalg.norm(self.x[-1]-self.x_F)


    def evolve_state(self, u):
        self.x.append(self.A @ self.x[-1] + self.B @ u)
        self.u.append(u)


    def orca_update(self, agent_list):
        # TODO urgent implement traditional orca
        u = self.find_u_orca(agent_list)
        self.evolve_state(u)


    def orca_mpc_update(self, N, agent_list):
        u = self.find_u_orca_mpc(N, agent_list)
        self.evolve_state(u)


    def add_orca_constraints(self, opti, x, idx, p_a, p_b, v_a, v_b):
        projection = projectOnVO((p_b-p_a)/self.tau, 2*self.radius/self.tau, v_a-v_b)
        # print("Projection is" + str(projection["projected_point"]), projection["region"])
        # abc = input("abc")
        region = projection["region"]
        u  = projection["projected_point"]-(v_a-v_b)
        is_outside = region == 1
        if is_outside:
            opti.subject_to((x[[idx],2:4].T-(v_a+u/2)).T@u<=0)
            pass
        else:
            #TODO this is a lazy hack
            opti.subject_to((x[[idx],2:4].T-(v_a+u/2)).T@u>=0)
            pass


    def find_u_orca_mpc(self, N, agent_list):
        n_x = self.A.shape[1]
        n_u = self.B.shape[1]
        opti = casadi.Opti()
        x = opti.variable(N+1,n_x)
        u = opti.variable(N,n_u)
        objective_sum = (x[[0],:].T-self.x_F).T @ self.Q @ (x[[0],:].T-self.x_F)
        opti.subject_to(x[[0],:].T == self.x[-1])
        x_cur_pred = self.x[-1]
        agent_cur_pred_list = [agent.x[len(self.x)-1] for agent in agent_list]
        vel_cur_pred = self.x[-1][2:4]
        for i in range(0,N):
            objective_sum += (x[[i+1],:].T-self.x_F ).T @ self.Q @ (x[[i+1],:].T-self.x_F) + u[[i],:] @ self.R @ u[[i],:].T
            print(self.B)
            opti.subject_to(x[[i+1],:].T == self.A @ x[[i],:].T + self.B @ u[[i],:].T)
            if not (self.H @ x[[i+1],:].T <= self.h).is_constant():
                opti.subject_to(self.H @ x[[i+1],:].T <= self.h)
            if not (self.G @ u[[i],:].T <= self.g).is_constant():
                opti.subject_to(self.G @ u[[i],:].T <= self.g)
            for agent in agent_list:
                if (agent._id == self._id):
                    continue
                #assumes agents are ordered by id (0 indexed)
                # EXPERIMENT
                p_a = x_cur_pred[0:2]
                p_b = agent_cur_pred_list[agent._id][0:2]
                v_a = x_cur_pred[2:4]
                v_b = agent_cur_pred_list[agent._id][2:4]
                # END EXPERIMENT
                self.add_orca_constraints(opti, x, i+1, p_a, p_b, v_a, v_b) # TODO add this for all neighbors and in general upgrade this
                agent_cur_pred_list[agent._id] = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])@agent_cur_pred_list[agent._id]
            x_cur_pred = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])@x_cur_pred
                
        
        opti.minimize(objective_sum)
        opts = {}
		opts["ipopt.linear_solver"] = 'ma27'
        opti.solver("ipopt",opts)
        sol = opti.solve()
        if N >1:
            return sol.value(u)[[0],:].T
        else:
            # print(sol.value(u))
            # print(sol.value(objective_sum))
            #DEBUG
            #EXPERIMENT
            return sol.value(u).reshape((n_u,1))

    def find_u_orca(self, agent_list):
        u = self.find_u_orca_mpc(1, agent_list)
        return u


    def plot_circles(self, x_list, y_list, radius):
        for i in range(len(x_list)):
            self.plot_circle(x_list[i], y_list[i], radius)
    

    def plot_circle(self, x, y, radius):
        theta = np.linspace(0,2*np.pi,10)
        x1 = x+radius*np.cos(theta)
        x2 = y+radius*np.sin(theta)
        plt.plot(x1,x2, color = self.color_list[self._id])


    def plot_trajectory(self):
        print(f'agent_id: {self._id}:')
        print("pos_x "+str([x_i[0,0] for x_i in self.x]))
        print("pos_y "+str([x_i[1,0] for x_i in self.x]))
        print("vel_x "+str([x_i[2,0] for x_i in self.x]))
        print("vel_y "+str([x_i[3,0] for x_i in self.x]))
        plt.plot([x_i[0,0] for x_i in self.x], [x_i[1,0] for x_i in self.x], label="Agent id: "+str(self._id))
        self.plot_circles([x_i[0,0] for x_i in self.x], [x_i[1,0] for x_i in self.x], self.radius)

