# -*- eval: (comment-tags-mode) -*-
import numpy as np
from . import stats


def chisqfn(p, fnc, x, y, cminv):
    r = fnc(x, p) - y
    return np.matmul(r, np.matmul(cminv, r.T))


def chisqfn2(p, fnc, x, y, cminv):
    r = fnc(x, *p) - y
    return np.matmul(r, np.matmul(cminv, r.T))


def chisqfn4(p, fnc, x, y, param, cminv):
    r = fnc(x, *p, *param) - y
    return np.matmul(r, np.matmul(cminv, r.T))


def chisqfn_bayes(p, fnc, x, y, cminv, priors, priorsigma):
    r = fnc(x, p) - y
    chiprior = np.sum((p - priors) ** 2 / priorsigma**2)
    return np.matmul(r, np.matmul(cminv, r.T)) + chiprior


def chisqfn3(p, fnc1, fnc2, fnc3, x1, x2, x3, y, cminv):
    """For three two-exponential functions"""
    r = (
        np.array(
            [
                *fnc1(x1, [p[2], p[0], p[3], p[1]]),
                *fnc2(x2, [p[4], p[0], p[5], p[1]]),
                *fnc3(x3, [p[6], p[0], p[7], p[1]]),
            ]
        )
        - y
    )
    return np.matmul(r, np.matmul(cminv, r.T))


def combchisq1exp(p, fnc1, fnc2, fnc3, x1, x2, x3, y, cminv):
    """For three one-exponential functions"""
    r = (
        np.array(
            [*fnc1(x1, [p[1], p[0]]), *fnc2(x2, [p[2], p[0]]), *fnc3(x3, [p[3], p[0]])]
        )
        - y
    )
    return np.matmul(r, np.matmul(cminv, r.T))


def combchisq2exp(p, fnc1, fnc2, x1, x2, y, cminv):
    """For two two-exponential functions with the same energies but different amplitudes"""
    r = (
        np.array(
            [*fnc1(x1, [p[2], p[0], p[3], p[1]]), *fnc2(x2, [p[4], p[0], p[5], p[1]])]
        )
        - y
    )
    return np.matmul(r, np.matmul(cminv, r.T))


def chisqfn1(p, fnc1, fnc2, fnc3, x1, x2, x3, y, cminv):
    r = (
        np.array(
            [*fnc1(x1, [p[1], p[0]]), *fnc2(x2, [p[2], p[0]]), *fnc3(x3, [p[3], p[0]])]
        )
        - y
    )
    return np.matmul(r, np.matmul(cminv, r.T))


def initffncs(fitflg, a=1.0):
    """initialize the functions"""
    if fitflg == "Aexp":
        fitfnc = Aexp()
    elif fitflg == "Twoexp":
        fitfnc = Twoexp()
    elif fitflg == "Twoexp_log":
        fitfnc = Twoexp_log()
    elif fitflg == "Threeexp":
        fitfnc = Threeexp()
    elif fitflg == "Threeexp_log":
        fitfnc = Threeexp_log()
    elif fitflg == "TwoexpRatio":
        fitfnc = TwoexpRatio()
    elif fitflg == "TwoexpRatio2":
        fitfnc = TwoexpRatio2()
    elif fitflg == "TwoexpRatio3":
        fitfnc = TwoexpRatio3()
    elif fitflg == "TwoexpRatio4":
        fitfnc = TwoexpRatio4()
    elif fitflg == "Constant":
        fitfnc = Constant()
    elif fitflg == "Linear":
        fitfnc = Linear()
    return fitfnc


############################################################
#
# single exponential f(x)=a*exp(b*x)


class Aexp:
    def __init__(self):
        self.npar = 2
        self.label = r"Aexp"
        self.initpar = np.array([1.0, 4.0e-1])
        self.bounds = [(-np.inf, np.inf), (-1, 1.0)]

    def initparfnc(self, data, timeslice=10):
        # Get the effective mass and amplitude for p0
        amp = stats.effamp(data)
        energy = stats.bs_effmass(data)
        # Set the initial guesses for the parameters
        self.initpar = np.array(
            [
                np.average(amp, axis=0)[timeslice],
                np.average(energy, axis=0)[timeslice],
            ]
        )
        self.bounds = np.array(
            [
                [
                    np.average(amp, axis=0)[timeslice]
                    - 10 * np.std(amp, axis=0)[timeslice],
                    np.average(amp, axis=0)[timeslice]
                    + 10 * np.std(amp, axis=0)[timeslice],
                ],
                [
                    np.average(energy, axis=0)[timeslice]
                    - 10 * np.std(energy, axis=0)[timeslice],
                    np.average(energy, axis=0)[timeslice]
                    + 10 * np.std(energy, axis=0)[timeslice],
                ],
            ]
        )
        self.priorsigma = np.array(
            [
                10 * np.std(amp, axis=0)[timeslice],
                10 * np.std(energy, axis=0)[timeslice],
            ]
        )

    def eval(self, x, p):
        """
        One-exponential function

        f(t,p) = p[0]*exp(-t*p[1])

        Amplitudes and energies:
        A_0 = p[0]
        E_0 = p[1]
        """
        return p[0] * np.exp(-1 * x * p[1])

    def eval_2(self, x, p0, p1):
        """
        One-exponential function

        f(t,p) = p0*exp(-t*p1)

        Amplitudes and energies:
        A_0 = p0
        E_0 = p1
        """
        return p0 * np.exp(-x * p1)


############################################################
#
# Two exponential f(x)=a*exp(b*x) + c*exp(d*x)


class Twoexp:
    def __init__(self):
        self.npar = 4
        self.label = r"Twoexp"
        self.initpar = np.array([1.0, 0.4, 1.0, 0.9])
        self.bounds = [(-np.inf, np.inf), (-1.0, 1.0), (-np.inf, np.inf), (-1.0, 3.0)]

    def initparfnc(self, data, timeslice=10):
        # Get the effective mass and amplitude for p0
        amp = stats.effamp(data, plot=False)
        energy = stats.bs_effmass(data, plot=False)
        # Set the initial guesses for the parameters
        self.initpar = np.array(
            [
                np.average(amp, axis=0)[timeslice],
                np.average(energy, axis=0)[timeslice],
                np.average(amp, axis=0)[timeslice],
                2 * np.average(energy, axis=0)[timeslice],
            ]
        )
        self.priorsigma = np.array(
            [
                10 * np.std(amp, axis=0)[timeslice],
                10 * np.std(energy, axis=0)[timeslice],
                10 * np.std(amp, axis=0)[timeslice],
                10 * np.std(energy, axis=0)[timeslice],
            ]
        )

    def eval(self, x, p):
        """
        Two-exponential function

        f(t,p) = p[0]*exp(-t*p[1]) + p[2]*exp(-t*p[3])

        Amplitudes and energies:
        A_0 = p[0]
        E_0 = p[1]
        A_1 = p[2]
        E_1 = p[3]
        """
        return p[0] * np.exp(-x * p[1]) + p[2] * np.exp(-x * p[3])

    def eval_2(self, x, p0, p1, p2, p3):
        """
        Two-exponential function

        f(t,p) = p0*exp(-t*p1) + p2*exp(-t*p3)

        Amplitudes and energies:
        A_0 = p0
        E_0 = p1
        A_1 = p2
        E_1 = p3
        """
        return p0 * np.exp(-x * p1) + p2 * np.exp(-x * p3)


############################################################
#
# Two exponential f(x)=a*exp(b*x) + c*exp(d*x)


class Twoexp_log:
    def __init__(self):
        self.npar = 4
        self.label = r"Twoexp_log"
        self.initpar = np.array([1.0, 0.4, 1.0, -0.9])
        self.bounds = [(-np.inf, np.inf), (-1.0, 1.0), (-np.inf, np.inf), (-1.0, 3.0)]

    def initparfnc(self, data, timeslice=10):
        # Get the effective mass and amplitude for p0
        amp = stats.effamp(data, plot=False)
        energy = stats.bs_effmass(data, plot=False)
        # Set the initial guesses for the parameters
        self.initpar = np.array(
            [
                np.average(amp, axis=0)[timeslice],
                np.average(energy, axis=0)[timeslice],
                1,
                np.log(0.8 * np.average(energy, axis=0)[timeslice]),
            ]
        )
        self.priorsigma = np.array(
            [
                2 * np.std(amp, axis=0)[timeslice],
                np.std(energy, axis=0)[timeslice],
                1,
                0.5,
            ]
        )

    def eval(self, x, p):
        """
        Two-exponential function where the excited states are defined as logarithms of the energy difference to fix the ordering of energies.

        f(t,p) = p[0]*exp(-t*p[1])*( 1 + p[2]*exp(-t*exp(p[3])))

        Amplitudes and energies:
        A_0 = p[0]
        E_0 = p[1]
        A_1 = p[0] * p[2]
        E_1 = p[1] + exp(p[3])

        Logarithmic parameters are from https://arxiv.org/abs/2009.11825
        """
        return p[0] * np.exp(-x * p[1]) * (1 + p[2] * np.exp(-x * np.exp(p[3])))

    def eval_2(self, x, p0, p1, p2, p3):
        """
        Two-exponential function where the excited states are defined as logarithms of the energy difference to fix the ordering of energies.

        f(t,p) = p0*exp(-t*p1)*( 1 + p2*exp(-t*exp(p3)))

        Amplitudes and energies:
        A_0 = p0
        E_0 = p1
        A_1 = p0 * p2
        E_1 = p1 + exp(p3)
        """
        return p0 * np.exp(-x * p1) * (1 + p2 * np.exp(-x * np.exp(p3)))


############################################################
#
# Three exponential f(x)=a*exp(b*x) + c*exp(d*x) + f*exp(g*x)
class Threeexp:
    def __init__(self):
        self.npar = 6
        self.label = r"Threeexp"
        self.initpar = np.array((0.08, 0.8, 0.08, 1.2, 0.08, 2.1))
        self.bounds = [
            (-np.inf, np.inf),
            (-1.0, 1.0),
            (-np.inf, np.inf),
            (-1.0, 3.0),
            (-np.inf, np.inf),
            (-1.0, 3.0),
        ]

    def initparfnc(self, data, timeslice=10):
        amp = stats.effamp(data, plot=False)
        energy = stats.bs_effmass(data, plot=False)
        self.initpar = np.array(
            [
                np.average(amp, axis=0)[timeslice],
                np.average(energy, axis=0)[timeslice],
                np.average(amp, axis=0)[timeslice],
                2 * np.log(np.average(energy, axis=0)[timeslice]),
                np.average(amp, axis=0)[timeslice],
                3 * np.log(np.average(energy, axis=0)[timeslice]),
            ]
        )
        self.priorsigma = np.array(
            [
                10 * np.std(amp, axis=0)[timeslice],
                10 * np.std(energy, axis=0)[timeslice],
                1,
                2,
                1,
                2,
            ]
        )

    def eval(self, x, p):
        """
        Three-exponential function

        f(t,p) = p[0]*exp(-t*p[1]) + p[2]*exp(-t*p[3]) + p[4]*exp(-t*p[5])

        Amplitudes and energies:
        A_0 = p[0]
        E_0 = p[1]
        A_1 = p[2]
        E_1 = p[3]
        A_2 = p[4]
        E_2 = p[5]
        """
        return (
            p[0] * np.exp(-x * p[1])
            + p[2] * np.exp(-x * p[3])
            + p[4] * np.exp(-x * p[5])
        )

    def eval_2(self, x, p0, p1, p2, p3, p4, p5):
        """
        Three-exponential function

        f(t,p) = p0*exp(-t*p1) + p2*exp(-t*p3) + p4*exp(-t*p5)

        Amplitudes and energies:
        A_0 = p0
        E_0 = p1
        A_1 = p2
        E_1 = p3
        A_2 = p4
        E_2 = p5
        """
        return p0 * np.exp(-x * p1) + p2 * np.exp(-x * p3) + p4 * np.exp(-x * p5)


############################################################
#
# Three exponential f(x)=a*exp(b*x) + c*exp(d*x) + f*exp(g*x)
class Threeexp_log:
    def __init__(self):
        self.npar = 6
        self.label = r"Threeexp_log"
        self.initpar = np.array((0.08, 0.8, 0.08, 1.2, 0.08, 2.1))
        self.bounds = [
            (-np.inf, np.inf),
            (-1.0, 1.0),
            (-np.inf, np.inf),
            (-1.0, 3.0),
            (-np.inf, np.inf),
            (-1.0, 3.0),
        ]

    def initparfnc(self, data, timeslice=13):
        amp = stats.effamp(data, plot=False)
        energy = stats.bs_effmass(data, plot=False)

        self.initpar = np.array(
            [
                np.average(amp, axis=0)[timeslice],
                np.average(energy, axis=0)[timeslice],
                1,
                np.log(0.25 * np.average(energy, axis=0)[timeslice]),
                1,
                np.log(0.25 * np.average(energy, axis=0)[timeslice]),
            ]
        )
        self.priorsigma = np.array(
            [
                2 * np.std(amp, axis=0)[timeslice],
                2 * np.std(energy, axis=0)[timeslice],
                1,
                1,
                1,
                2,
            ]
        )

    def eval(self, x, p):
        """
        Three-exponential function where the excited states are defined as logarithms of the energy difference to fix the ordering of energies.

        f(t,p) = p[0]*exp(-t*p[1])*( 1 + p[2]*exp(-t*exp(p[3]))  + p[4]*exp(-t*(exp(p[3]) + exp(p[5]))))

        Amplitudes and energies:
        A_0 = p[0]
        E_0 = p[1]
        A_1 = p[0] * p[2]
        E_1 = p[1] + exp(p[3])
        A_2 = p[0] * p[4]
        E_2 = p[1] + exp(p[3]) + exp(p[5])
        """
        return (
            p[0]
            * np.exp(-x * p[1])
            * (
                1
                + p[2] * np.exp(-x * np.exp(p[3]))
                + p[4] * np.exp(-x * (np.exp(p[3]) + np.exp(p[5])))
            )
        )

    def eval_2(self, x, p0, p1, p2, p3, p4, p5):
        """
        Three-exponential function where the excited states are defined as logarithms of the energy difference to fix the ordering of energies.

        f(t,p) = p0*exp(-t*p1)*( 1 + p2*exp(-t*exp(p3))  + p4*exp(-t*(exp(p3) + exp(p5))))

        Amplitudes and energies:
        A_0 = p0
        E_0 = p1
        A_1 = p0 * p2
        E_1 = p1 + exp(p3)
        A_2 = p0 * p4
        E_2 = p1 + exp(p3) + exp(p5)
        """
        return p0 * np.exp(-x * p1)(
            1
            + p2 * np.exp(-x * np.exp(p3))
            + p4 * np.exp(-x * (np.exp(p3) + np.exp(p5)))
        )


############################################################
#
# Two exponential ratio
#
class TwoexpRatio:
    def __init__(self):
        self.npar = 2
        self.label = r"TwoexpRatio"
        # self.initpar=np.array([0.08,1.8e-6,0.8,8.4e-1]) #p+2+2+0
        # self.initpar=np.array([1.,1.15e-4,-1.4,1.4]) #p+0+0+0
        self.initpar = np.array([1.8e-4, 3.5e-4])  # p+1+0+0
        # self.bounds=([-np.inf,0,-np.inf,0],[np.inf,10,np.inf,10])
        self.q = [1.0, 1.0]
        # q[0] = A1/A0
        # q[1] = E_1 - E_0
        self.bounds = [(-1.0, 1.0), (-1.0, 3.0)]
        # print("Initialising Two exp fitter")

    def initparfnc(self, y, i=0):
        # self.initpar=np.array([1.8e-4,3.5e-4]) #p+1+0+0
        # self.initpar=np.array([1.,1.8e-4,-3.8e-4,3.5e-1]) #p+1+0+0
        pass

    def eval(self, x, p):
        """evaluate"""
        # return (np.exp(-x*p[0])+p[1]*np.exp(-x*(p[2]+p[3])))/(1+p[1]*np.exp(-x*p[2]))
        return (np.exp(-x * p[0]) + self.q[0] * np.exp(-x * (self.q[1] + p[1]))) / (
            1 + self.q[0] * np.exp(-x * self.q[1])
        )

    def eval2(self, x, q, p):
        """evaluate"""
        # return (np.exp(-x*p[0])+p[1]*np.exp(-x*(p[2]+p[3])))/(1+p[1]*np.exp(-x*p[2]))
        return (np.exp(-x * p[0]) + q[0] * np.exp(-x * (q[1] + p[1]))) / (
            1 + q[0] * np.exp(-x * q[1])
        )


############################################################
#
# Two exponential ratio #2
# With lambda independant amplitudes
class TwoexpRatio2:
    def __init__(self):
        self.npar = 6
        self.label = r"TwoexpRatio2"
        # self.initpar=np.array([0.08,1.8e-6,0.8,8.4e-1]) #p+2+2+0
        # self.initpar=np.array([1.,1.15e-4,-1.4,1.4]) #p+0+0+0
        self.initpar = np.array([1.0, 1.0, 0.45, 0.9, 1.8e-4, 3.5e-4])  # p+1+0+0
        # self.bounds=([-np.inf,0,-np.inf,0],[np.inf,10,np.inf,10])
        self.bounds = [
            (-np.inf, np.inf),
            (-np.inf, np.inf),
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, 3.0),
            (-1.0, 3.0),
        ]
        # self.bounds=[(-1.,1.),(-1.,3.)]
        # print("Initialising Two exp fitter")

    def initparfnc(self, y, i=0):
        # self.initpar=np.array([1.8e-4,3.5e-4]) #p+1+0+0
        # self.initpar=np.array([1.,1.8e-4,-3.8e-4,3.5e-1]) #p+1+0+0
        self.initpar[0] = y[0]

    def eval(self, x, p):
        """evaluate"""
        # return (np.exp(-x*p[0])+p[1]*np.exp(-x*(p[2]+p[3])))/(1+p[1]*np.exp(-x*p[2]))
        # return (np.exp(-x*p[0])+self.q[0]*np.exp(-x*(self.q[1]+p[1])))/(1+self.q[0]*np.exp(-x*self.q[1]))
        return (
            p[0] * np.exp(-x * (p[2] + p[4])) + p[1] * np.exp(-x * (p[3] + p[5]))
        ) / (p[0] * np.exp(-x * p[2]) + p[1] * np.exp(-x * p[3]))

    def eval2(self, x, q, p):
        """evaluate"""
        # return (np.exp(-x*p[0])+p[1]*np.exp(-x*(p[2]+p[3])))/(1+p[1]*np.exp(-x*p[2]))
        return (np.exp(-x * p[0]) + q[0] * np.exp(-x * (q[1] + p[1]))) / (
            1 + q[0] * np.exp(-x * q[1])
        )


############################################################
#
# Two exponential ratio #3
#
class TwoexpRatio3:
    def __init__(self):
        self.npar = 4
        self.label = r"TwoexpRatio3"
        # self.initpar=np.array([0.08,1.8e-6,0.8,8.4e-1]) #p+2+2+0
        # self.initpar=np.array([1.,1.15e-4,-1.4,1.4]) #p+0+0+0
        self.initpar = np.array([1.0, 1.8e-4, 1.0, 3.5e-4])  # p+1+0+0
        # self.bounds=([-np.inf,0,-np.inf,0],[np.inf,10,np.inf,10])
        self.q = np.array([1.0, 1.0, 1.0])
        # q[0] = A0
        # q[1] = A1
        # q[2] = E_1 - E_0
        self.bounds = np.array(
            [(-np.inf, np.inf), (-1.0, 1.0), (-np.inf, np.inf), (-1.0, 3.0)]
        )
        # print("Initialising Two exp ratio fitter")

    def initparfnc(self, y, i=0):
        # self.initpar=np.array([1.8e-4,3.5e-4]) #p+1+0+0
        # self.initpar=np.array([1.,1.8e-4,-3.8e-4,3.5e-1]) #p+1+0+0
        pass

    def eval(self, x, p):
        """evaluate"""
        # return (np.exp(-x*p[0])+p[1]*np.exp(-x*(p[2]+p[3])))/(1+p[1]*np.exp(-x*p[2]))
        # return (np.exp(-x*p[0])+self.q[0]*np.exp(-x*(self.q[1]+p[1])))/(1+self.q[0]*np.exp(-x*self.q[1]))
        return (p[0] * np.exp(-x * p[1]) + p[2] * np.exp(-x * (self.q[2] + p[3]))) / (
            self.q[0] + self.q[1] * np.exp(-x * self.q[2])
        )

    def eval2(self, x, q, p):
        """evaluate"""
        # return (np.exp(-x*p[0])+p[1]*np.exp(-x*(p[2]+p[3])))/(1+p[1]*np.exp(-x*p[2]))
        return (p[0] * np.exp(-x * p[1]) + p[2] * np.exp(-x * (q[2] + p[3]))) / (
            q[0] + q[1] * np.exp(-x * q[2])
        )


############################################################
#
# Two exponential ratio #4
#
class TwoexpRatio4:
    def __init__(self):
        self.npar = 4
        self.label = r"TwoexpRatio4"
        self.initpar = np.array([1.0e-4, 1.8e-4, 1.0e-4, 3.5e-4])  # p+1+0+0
        self.q = np.array([1.0, 1.0, 1.0, 1.0])
        # q[0] = A0
        # q[1] = E_0
        # q[2] = A1
        # q[3] = E_1
        self.bounds = np.array(
            [(-np.inf, np.inf), (-1.0, 1.0), (-np.inf, np.inf), (-1.0, 3.0)]
        )
        # print("Initialising Two exp ratio fitter")

    def initparfnc(self, data):
        # Get the effective mass and amplitude for p0
        amp = stats.effamp(data, plot=False)
        energy = stats.bs_effmass(data, plot=False)
        # Set the initial guesses for the parameters
        timeslice = 0
        # self.initpar = [
        #     amp[timeslice][0],
        #     energy[timeslice][0],
        #     amp[timeslice][0],
        #     energy[timeslice][0],
        # ]
        self.initpar = [
            1,
            energy[timeslice][0],
            1,
            energy[timeslice][0],
        ]
        self.bounds = [
            [
                amp[timeslice][0] - amp[timeslice][1],
                amp[timeslice][0] + amp[timeslice][1],
            ],
            [
                energy[timeslice][0] - energy[timeslice][1],
                energy[timeslice][0] + energy[timeslice][1],
            ],
            [
                1.5 * amp[timeslice][0] - amp[timeslice][1],
                1.5 * amp[timeslice][0] + amp[timeslice][1],
            ],
            [
                1.5 * energy[timeslice][0] - energy[timeslice][1],
                1.5 * energy[timeslice][0] + energy[timeslice][1],
            ],
        ]

    def eval(self, x, p):
        """evaluate"""
        return (
            (self.q[0] + p[0]) * np.exp(-x * (self.q[1] + p[1]))
            + (self.q[2] + p[2]) * np.exp(-x * (self.q[3] + p[3]))
        ) / (
            (self.q[0] - p[0]) * np.exp(-x * (self.q[1] - p[1]))
            + (self.q[2] - p[2]) * np.exp(-x * (self.q[3] - p[3]))
        )

    def eval2(self, x, q, p):
        """evaluate the function with the q variables being the amplitudes and energies"""
        return (
            (q[0] + p[0]) * np.exp(-x * (q[1] + p[1]))
            + (q[2] + p[2]) * np.exp(-x * (q[3] + p[3]))
        ) / (
            (q[0] - p[0]) * np.exp(-x * (q[1] - p[1]))
            + (q[2] - p[2]) * np.exp(-x * (q[3] - p[3]))
        )


############################################################
#
# Two exponential ratio #4
#
class TwoexpRatio5:
    def __init__(self):
        self.npar = 4
        self.label = r"TwoexpRatio5"
        self.initpar = np.array([1.0e-4, 1.8e-4, 1.0e-4, 3.5e-4])  # p+1+0+0
        self.q = np.array([1.0, 1.0, 1.0, 1.0])
        # q[0] = A0
        # q[1] = E_0
        # q[2] = A1
        # q[3] = E_1
        self.bounds = np.array(
            [(-np.inf, np.inf), (-1.0, 1.0), (-np.inf, np.inf), (-1.0, 3.0)]
        )
        # print("Initialising Two exp ratio fitter")

    def initparfnc(self, data):
        # Get the effective mass and amplitude for p0
        amp = stats.effamp(data, plot=False)
        energy = stats.bs_effmass(data, plot=False)
        # Set the initial guesses for the parameters
        timeslice = 0
        # self.initpar = [
        #     amp[timeslice][0],
        #     energy[timeslice][0],
        #     amp[timeslice][0],
        #     energy[timeslice][0],
        # ]
        self.initpar = [
            1,
            energy[timeslice][0],
            1,
            energy[timeslice][0],
        ]
        self.bounds = [
            [
                amp[timeslice][0] - amp[timeslice][1],
                amp[timeslice][0] + amp[timeslice][1],
            ],
            [
                energy[timeslice][0] - energy[timeslice][1],
                energy[timeslice][0] + energy[timeslice][1],
            ],
            [
                1.5 * amp[timeslice][0] - amp[timeslice][1],
                1.5 * amp[timeslice][0] + amp[timeslice][1],
            ],
            [
                1.5 * energy[timeslice][0] - energy[timeslice][1],
                1.5 * energy[timeslice][0] + energy[timeslice][1],
            ],
        ]

    def eval(self, x, p):
        """evaluate"""
        return (
            (self.q[0] + p[0]) * np.exp(-x * (self.q[1] + p[1]))
            + (self.q[2] + p[2]) * np.exp(-x * (self.q[3] + p[3]))
        ) / (
            (self.q[0] - p[0]) * np.exp(-x * (self.q[1] - p[1]))
            + (self.q[2] - p[2]) * np.exp(-x * (self.q[3] - p[3]))
        )

    def eval2(self, x, q, p):
        """evaluate the function with the q variables being the amplitudes and energies"""
        return (
            (q[0] + p[0]) * np.exp(-x * (q[1] + p[1]))
            + (q[2] + p[2]) * np.exp(-x * (q[3] + p[3]))
        ) / (
            (q[0] - p[0]) * np.exp(-x * (q[1] - p[1]))
            + (q[2] - p[2]) * np.exp(-x * (q[3] - p[3]))
        )


############################################################
#
# Combined Two exponential f(x)=a0*exp(b*x) + c0*exp(d*x), g(x)=a1*exp(b*x) + c1*exp(d*x)


class Combtwoexp:
    def __init__(self):
        self.npar = 4
        self.label = r"Combtwoexp"
        self.initpar = np.array([0.08, 0.5, 0.08, 0.8, 0.08, 0.08])
        self.bounds = ([-np.inf, 0, -np.inf, 0.5], [np.inf, 1, np.inf, 10])
        # print("Initialising Combined Two exp fitter")

    def initparfnc(self, y, i=0):
        self.initpar = np.array([0.08, 0.5, 0.08, 0.8, 0.08, 0.08])
        self.initpar[i] = y[0]
        self.initpar[2] = y[0]
        # self.initpar[2]=y[2]

    def eval(self, x, p):
        """evaluate"""
        return p[0] * np.exp(-x * p[1]) + p[2] * np.exp(-x * p[3])

    def eval_2(self, x, p0, p1, p2, p3):
        """evaluate"""
        return p0 * np.exp(-x * p1) + p2 * np.exp(-x * p3)

    def dereval(self, x, p, pe, v):
        """evaluate the derivative"""
        eA = np.exp(-x * p[1])
        eC = np.exp(-x * p[3])
        eB = -x * p[0] * eA
        eD = -x * p[2] * eC
        return np.sqrt(
            pe[0] ** 2 * eA**2
            + 2 * pe[0] * pe[1] * v[0] * eA * eB
            + 2 * pe[0] * pe[2] * v[1] * eA * eC
            + 2 * pe[0] * pe[3] * v[2] * eA * eD
            + pe[1] ** 2 * eB**2
            + 2 * pe[1] * pe[2] * v[3] * eB * eC
            + 2 * pe[1] * pe[3] * v[4] * eB * eD
            + pe[2] ** 2 * eC**2
            + 2 * pe[2] * pe[3] * v[5] * eC * eD
            + pe[3] ** 2 * eD**2
        )


############################################################
#
# constant function f(x)=a
class Constant:
    def __init__(self):
        self.npar = 1
        self.label = r"constant"
        self.initpar = np.array([1.0])
        self.bounds = [(-np.inf, np.inf)]
        # print("Initialising Constant fitter")

    def initparfnc(self, data, timeslice=10):
        self.initpar = np.array([np.average(data, axis=0)[timeslice]])
        self.bounds = np.array(
            [
                np.average(data, axis=0)[timeslice]
                - 10 * np.std(data, axis=0)[timeslice],
                np.average(data, axis=0)[timeslice]
                + 10 * np.std(data, axis=0)[timeslice],
            ]
        )
        self.priorsigma = np.array([10 * np.std(data, axis=0)[timeslice]])

    def eval(self, x, p):
        """evaluate"""
        return p[0] * x / x


############################################################
#
# Linear function f(x)=ax+b
class Linear:
    def __init__(self):
        self.npar = 2
        self.label = r"linear"
        self.initpar = np.array([1.0, 1.0])
        self.bounds = [(-np.inf, np.inf), (-np.inf, np.inf)]

    def initparfnc(self, data, timeslice=10):
        slope = data[:, timeslice + 1] - data[:, timeslice]
        print("slope = ", slope)
        yintercept = data[:, timeslice] - timeslice * slope

        self.initpar = np.array([np.average(slope), np.average(yintercept)])
        self.bounds = np.array(
            [
                [
                    np.average(slope) - np.std(slope),
                    np.average(slope) + np.std(slope),
                ],
                [
                    np.average(yintercept) - np.std(yintercept),
                    np.average(yintercept) + np.std(yintercept),
                ],
            ]
        )
        self.priorsigma = np.array([10 * np.std(slope), 10 * np.std(yintercept)])

    def eval(self, x, p):
        """evaluate"""
        return p[0] * x + p[1]

    def eval_2(self, x, p0, p1):
        """evaluate"""
        return p0 * x + p1


def constant(x, p0):
    # return p0 * x / x
    return p0 * np.ones(len(x))


def linear(x, p0, p1):
    return p0 + p1 * x


def oneexp(t, A0, m0):
    return A0 * np.exp(-m0 * t)


def twoexp(t, A0, A1, m0, m1):
    return A0 * np.exp(-m0 * t) + A1 * np.exp(-m1 * t)


def twoexp_cov(t, p):
    return p[0] * np.exp(-p[2] * t) + p[1] * np.exp(-p[3] * t)


def threeexp(t, A0, A1, A2, m0, m1, m2):
    return A0 * np.exp(-m0 * t) + A1 * np.exp(-m1 * t) + A2 * np.exp(-m2 * t)


def comb1exp3sink(t, A0_pt, A1_pt, A0_30, A0_60, m0):
    G = np.array([])
    G = np.append(G, np.array(A0_pt * np.exp(-m0 * t[0])))
    G = np.append(G, np.array(A0_30 * np.exp(-m0 * t[1])))
    G = np.append(G, np.array(A0_60 * np.exp(-m0 * t[2])))
    return G


def comb2exp3sink(t, A0_pt, A1_pt, A0_30, A1_30, A0_60, A1_60, m0, m1):
    G = np.array([])
    G = np.append(G, np.array(A0_pt * np.exp(-m0 * t[0]) + A1_pt * np.exp(-m1 * t[0])))
    G = np.append(G, np.array(A0_30 * np.exp(-m0 * t[1]) + A1_30 * np.exp(-m1 * t[1])))
    G = np.append(G, np.array(A0_60 * np.exp(-m0 * t[2]) + A1_60 * np.exp(-m1 * t[2])))
    return G


def comb2exp3sink_test(x, A0_pt, A1_pt, A0_30, A1_30, A0_60, A1_60, m0, m1):
    t = x[0]
    f = x[1]
    G = np.array([])
    G = np.append(
        G, np.array(A0_pt * np.exp(-m0 * t[: f[0]]) + A1_pt * np.exp(-m1 * t[: f[0]]))
    )
    G = np.append(
        G,
        np.array(
            A0_30 * np.exp(-m0 * t[f[0] : f[1]]) + A1_30 * np.exp(-m1 * t[f[0] : f[1]])
        ),
    )
    G = np.append(
        G,
        np.array(
            A0_60 * np.exp(-m0 * t[f[1] : f[2]]) + A1_60 * np.exp(-m1 * t[f[1] : f[2]])
        ),
    )
    return G


def comb3exp3sink(
    t, A0_pt, A1_pt, A2_pt, A0_30, A1_30, A2_30, A0_60, A1_60, A2_60, m0, m1, m2
):
    G = np.array([])
    G = np.append(
        G,
        np.array(
            A0_pt * np.exp(-m0 * t[0])
            + A1_pt * np.exp(-m1 * t[0])
            + A2_pt * np.exp(-m2 * t[0])
        ),
    )
    G = np.append(
        G,
        np.array(
            A0_30 * np.exp(-m0 * t[1])
            + A1_30 * np.exp(-m1 * t[1])
            + A2_60 * np.exp(-m2 * t[1])
        ),
    )
    G = np.append(
        G,
        np.array(
            A0_60 * np.exp(-m0 * t[2])
            + A1_60 * np.exp(-m1 * t[2])
            + A2_60 * np.exp(-m2 * t[2])
        ),
    )
    return G
