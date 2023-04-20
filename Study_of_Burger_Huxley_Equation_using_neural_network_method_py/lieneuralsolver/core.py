# Importing the necessary packages
import autograd.numpy as np
from autograd import grad
from autograd import elementwise_grad as egrad
import autograd.numpy.random as npr
from autograd.misc.flatten import flatten
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt


#
count = 0

# Initialize neural network parameters
def init_weights(n_in=1, n_hidden=10, n_out=1):
    W1 = npr.randn(n_in, n_hidden)
    b1 = np.zeros(n_hidden)
    W2 = npr.randn(n_hidden, n_out)
    b2 = np.zeros(n_out)
    params = [W1, b1, W2, b2]
    return params

# Construct \hat{u}_1 neural network solution
def predict1(params, t, act=np.tanh):
    W1, b1, W2, b2 = params
    a = act(np.dot(t, W1) + b1)
    out = np.dot(a, W2) + b2
    y0_1lie = -(1/12)*np.exp(t/2)*(-7+np.exp((3*t)/2))
    # Huxley equation   (3/4)-(1/4)*np.exp(-t/sqrt(2))                  # D_1 relative solution \bar{u}_{1}
    # Burger_Huxley_Equation  (1/12)*(7-np.exp((3*t)/2))                # D_1 relative solution \bar{u}_{1}
    # Burger_Fisher Equation  -(1/12)*np.exp(t/2)*(-7+np.exp((3*t)/2))  # D_1 relative solution \bar{u}_{1}
    y = y0_1lie + t*out
    return y

# Construct \hat{u}_2 neural network solution
def predict2(params, t, act=np.tanh):
    W1, b1, W2, b2 = params
    a = act(np.dot(t, W1) + b1)
    out = np.dot(a, W2) + b2
    y0_2lie = ((1/24)*(7*np.exp(t/2)))-((np.exp(2*t))/6)
    # Huxley equation   (np.exp(-t/sqrt(2)))/(4*sqrt(2))                   # D_1 relative solution \bar{u}_{2}
    # Burger_Huxley_Equation  -(1/8)*(np.exp((3*t)/2))                     # D_1 relative solution \bar{u}_{2}
    # Burger_Fisher Equation  ((1/24)*(7*np.exp(t/2)))-((np.exp(2*t))/6)   # D_1 relative solution \bar{u}_{2}
    y = y0_2lie + t*out
    return y

# Ignore predict3, used when the number of equations is 3.
def predict3(params, t, act=np.tanh):
    W1, b1, W2, b2 = params
    a = act(np.dot(t, W1) + b1)
    out = np.dot(a, W2) + b2
    y0_3lie = -4*np.cos(2*t)
    y = y0_3lie + t*out
    return y

predict1_dt = egrad(predict1, argnum=1)   #\hat{u}_1/dt   t is \xi

predict2_dt = egrad(predict2, argnum=1)   #\hat{u}_2/dt   t is \xi

predict3_dt = egrad(predict3, argnum=1)   #igone

# NNSolver
class NNSolver(object):
    def __init__(self, f, t, y0_list, n_hidden=10):
        Nvar = len(y0_list)
        assert len(f(t[0], y0_list)) == Nvar,\

        assert t.shape == (t.size, 1),

        self.Nvar = Nvar
        self.f = f
        self.t = t
        self.y0_list = y0_list
        self.n_hidden = n_hidden
        self.loss = 0
        self.reset_weights()

    def __str__(self):
        return ('Neural ODE Solver \n'
                'Number of equations:       {} \n'      # Number of output equations
                'Initial condition y0:      {} \n'      # initial values
                'Numnber of hidden units:   {} \n'      # number of neurons in the hidden layer
                'Number of training points: {} '        # number of training points
                .format(self.Nvar, self.y0_list, self.n_hidden, self.t.size)
                )

    def __repr__(self):
        return self.__str__()

    def reset_weights(self):  # Resetting neural network parameters
        self.params_list = [init_weights(n_hidden=self.n_hidden)
                            for _ in range(self.Nvar)]
        flattened_params, unflat_func = flatten(self.params_list)
        self.flattened_params = flattened_params
        self.unflat_func = unflat_func

    def loss_func(self, params_list):  # Define the loss function
        y0_list = self.y0_list
        t = self.t
        f = self.f

        if len(params_list) == 1:
            y_pred_list = [predict1(params_list[0], t)]
            dydt_pred_list = [predict1_dt(params_list[0], t)]
        elif len(params_list) == 2:
            y_pred_list = [predict1(params_list[0], t), predict2(params_list[1], t)]
            dydt_pred_list = [predict1_dt(params_list[0], t), predict2_dt(params_list[1], t)]
        else:
            y_pred_list = [predict1(params_list[0], t), predict2(params_list[1], t), predict3(params_list[2], t)]
            dydt_pred_list = [predict1_dt(params_list[0], t), predict2_dt(params_list[1], t), predict3_dt(params_list[2], t)]
        f_pred_list = f(t, y_pred_list)

        loss_total = 0.0
        for f_pred, dydt_pred in zip(f_pred_list, dydt_pred_list):
            pre_1= predict1(params_list[0], -5).flatten()    #L_I of \hat{u}_1
            pre_2= predict2(params_list[1], -5).flatten()    #L_I of \hat{u}_2
            loss = np.mean((dydt_pred-f_pred)**2)+(1/2)*(pre_1-0.0758582)**2+(1/2)*(pre_2-0.0350519)**2  #L_total
            loss_total += loss
        return loss_total

# Give the definition of the loss function for the following equation
    # Burger Huxley Equation(-5)  +(1/2)*(predict1(params_list[0], -5)-0.924142)**2+(1/2)*(predict2(params_list[1], -5)+0.0350519)**2
    # Burger Huxley Equation(-5)  np.mean((dydt_pred-f_pred)**2)+(1/2)*(pre_1-0.924142)**2+(1/2)*(pre_2+0.0350519)**2
    # Burger Fisher Equation (-5) +(1/2)*(pre_1-0.0758582)**2+(1/2)*(pre_2-0.0350519)**2
    # Burger Fisher Equation (-5) np.mean((dydt_pred-f_pred)**2)+(1/2)*(pre_1-0.0758582)**2+(1/2)*(pre_2-0.0350519)**2
    # Huxley Equation (-2)  +(1/2)*(pre_1-0.19557)**2+(1/2)*(pre_2-0.111244)**2
    # Huxley Equation (-2)  np.mean((dydt_pred-f_pred)**2)+(1/2)*(pre_1-0.19557)**2+(1/2)*(pre_2-0.111244)**2

    def loss_wrap(self, flattened_params):
        params_list = self.unflat_func(flattened_params)
        return self.loss_func(params_list)

# Training process  BFGS optimization algorithm
    def train(self, method='BFGS', maxiter=2000, iprint=200):
        self.x = []

        global count
        global loss_arr
        count = 0
        loss_arr = []
        def print_loss(x):
            global count
            if count % iprint == 0:
                print("iteration:", count, "loss: ", self.loss_wrap(x))

            count += 1
            self.x.append(self.unflat_func(x))
            loss_arr.append(self.loss_wrap(x))

        opt = minimize(self.loss_wrap, x0=self.flattened_params,
                       jac=grad(self.loss_wrap), method=method,
                       callback=print_loss,
                       options={'disp': True, 'maxiter': maxiter})

        self.loss = loss_arr
        self.flattened_params = opt.x
        self.params_list = self.unflat_func(opt.x)


# Define the network solution \hat{u}
    def predict(self, t=None, params_list=None):
        if t is None:
            t = self.t

        if params_list is None:
            if len(self.params_list) == 1:
                y_pred_list=[predict1(self.params_list[0], t).squeeze()]
                dydt_pred_list = [predict1_dt(self.params_list[0], t).squeeze()]

            elif len(self.params_list) == 2:
                y_pred_list = [predict1(self.params_list[0], t).squeeze(), predict2(self.params_list[1], t).squeeze()]
                dydt_pred_list = [predict1_dt(self.params_list[0], t).squeeze(), predict2_dt(self.params_list[1], t).squeeze()]

            else:
                y_pred_list = [predict1(self.params_list[0], t).squeeze(), predict2(self.params_list[1], t).squeeze(), predict3(self.params_list[2], t).squeeze()]
                dydt_pred_list = [predict1_dt(self.params_list[0], t).squeeze(), predict2_dt(self.params_list[1], t).squeeze(),
                                  predict3_dt(self.params_list[2], t).squeeze()]

            return y_pred_list, dydt_pred_list


        else:
            if len(params_list) == 1:
                y_pred_list = [predict1(params_list[0], t).squeeze()]

            elif len(params_list) == 2:
                y_pred_list = [predict1(params_list[0], t).squeeze(), predict2(params_list[1], t).squeeze()]

            else:
                y_pred_list = [predict1(params_list[0], t).squeeze(), predict2(params_list[1], t).squeeze(),
                               predict3(params_list[2], t).squeeze()]
            return y_pred_list

# Animation Show
    def result(self,t = None, anim = False, interval = 50, every_n_iter = 1):
        if t is None:
            t = self.t


        if anim:
            y_train = np.array([self.predict(t = t.reshape(-1,1), params_list=x) for x in self.x])[::-every_n_iter][::-1]
            n = y_train.shape[1]

            fig, ax = plt.subplots(1,2,figsize = (16,6))

            sol_cont = solve_ivp(self.f, [t.min(), t.max()], self.y0_list, method='Radau', rtol=1e-5)

            sol_dis = solve_ivp(self.f, t_span = [t.min(), t.max()], t_eval=t, y0 = self.y0_list, method='Radau', rtol=1e-5)

            y_diff = np.array([np.array(sol_dis.y[i]) for i in range(n)])

            ax[0].set_xlim((t[0], t[-1]))
            ax[0].set_ylim((np.min(y_train[-1,:,:]), np.max(y_train[-1,:,:])))
            ax[1].set_xlim((t[0], t[-1]))
            ax[1].set_ylim((np.min(y_train[-1,:,:] - y_diff), np.max(y_train[-1,:,:] - y_diff)))
            for i in range(n):
                ax[0].plot(sol_cont.t, sol_cont.y[i], label='y{}'.format(i+1))

            scatters_pred = []
            scatters_diff = []
            for i in range(n):

                scat1 = ax[0].scatter([], [], label = 'y_pred{}'.format(i+1))
                scatters_pred.append(scat1)

                scat2 = ax[1].scatter([], [], label = 'y{}'.format(i+1))
                scatters_diff.append(scat2)

            ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, ncol=4)
            ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, ncol=4)

            ax[0].set_title("NN Prediction and Truth", fontsize = 18)
            ax[1].set_title("NN Prediction - Truth", fontsize = 18)

            def init():
                for k in range(n):
                    scatters_pred[k].set_offsets(np.hstack(([], [])))
                    scatters_diff[k].set_offsets(np.hstack(([], [])))
                return (scatters_pred + scatters_diff)


            def animate(i):
                for k in range(n):
                    scatters_pred[k].set_offsets(np.c_[t, y_train[i,k,:]])

                    scatters_diff[k].set_offsets(np.c_[t, y_train[i,k,:] - y_diff[k]])


            anim_pred = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=y_train.shape[0], interval=interval, blit=True)

            return anim_pred

# Define the loss function and iteration count image
    def plot_loss(self):
        plt.figure(figsize=(8,6))
        plt.semilogy(range(len(loss_arr)), loss_arr, label="BFGS")
        plt.legend(loc='best')
        plt.xlabel("Training Iterations")
        plt.ylabel("Log Loss")
        plt.show()
