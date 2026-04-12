from math import isnan, sin, cos
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

plt.style.use('bmh')

muE = 398.6004415
sigma = 1.e0              # баллистический коэффициент, м^2/кг
tau = 60.e-3               # шаг интегриррования по времени 60 сек (в тыс. сек)
omega0 = 7.292115085e-2   # рад в тыс. сек, угловая скорость вращения Земли

omega_vec = np.array([0., 0., omega0])

rho_data = [
    [150,   180, 2.070e-9,  22.523],
    [180,   200, 5.464e-10, 29.740],
    [200,   250, 2.789e-10, 37.105],
    [250,   300, 7.248e-11, 45.546],
    [300,   350, 2.418e-11, 53.628],
    [350,   400, 9.518e-12, 53.298],
    [400,   450, 3.725e-12, 58.515],
    [450,   500, 1.585e-12, 60.828],
    [500,   600, 6.967e-13, 63.822],
    [600,   700, 1.454e-13, 71.835],
    [700,   800, 3.614e-14, 88.667],
    [800,   900, 1.170e-14, 124.64],
    [900,  1000, 5.245e-15, 181.05],
    [1000, 1500, 3.019e-15, 268.00]
]
rho_array = np.array(rho_data)

def calc_rho_atm(h):
    """Рассчитывает плотность атомсферы Земли на высоте h.
    Args:
       h: высота над эллипсоидом, тыс. км

    """
    h_km = h * 1.e3
    if h_km > rho_array[-1, 1]:
        return 0.0
    
    h_index = np.searchsorted(rho_array[:, 0], h_km) - 1
    if h_index < 0:
        h_index = 0
    
    h0, _, rho0, H = rho_array[h_index, :]

    rho = rho0 * np.exp( - (h_km - h0) / H)

    return rho


def rk4(func, t0, x0, h):
    """RK4 Интегрирование

    """

    k1 = func(t0, x0)
    k2 = func(t0 + h * 0.5, x0 + k1 * h * 0.5)
    k3 = func(t0 + h * 0.5, x0 + k2 * h * 0.5)
    k4 = func(t0 + h, x0 + k3 * h)

    return x0 + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def solve_kepler(e: float, M: float, tol: float = 1.e-6):
    E = M % (2 * np.pi)
    E_prev = 0.
    first = True
    i = 0
    while first or abs(E_prev - E) > tol:
        E_prev = E
        E = E + (M - E + e * sin(E)) / (1 - e * cos(E))
        i += 1
        if i >= 10:
            print(M, e)
            print(f"Предел по итерациям. Последняя разность {abs(E - E_prev)}")
            break
        
        if first:
            first = False
        
    return E


def elems2state(elem: np.ndarray, mu: float = muE) -> np.ndarray:
    """Преобразует элементы орбиты a, e, i, Omega, omega, M в вектор
    состояния

    """
    
    a, e, i, Omega, omega, M = elem.tolist()
    P, Q = pq_from_elems(i, Omega, omega)
    state = np.zeros(6)
    E = solve_kepler(e, M)
    nu = 2. * np.arctan(np.tan(E * 0.5) * np.sqrt((1. + e) / (1. - e)))
    p = a * (1. - e * e)
    r = p / (1 + e * np.cos(nu))
    x = r * np.cos(nu)
    y = r * np.sin(nu)
    v_coeff = np.sqrt(mu / p)
    x_dot = -v_coeff * np.sin(nu)
    y_dot = v_coeff * (e + np.cos(nu))

    state[:3] = P * x + Q * y
    state[3:] = P * x_dot + Q * y_dot

    return state


def state2elems(state: np.ndarray, mu: float = muE) -> np.ndarray:
    """Преобразует вектор состояния (тыс. км, км/с) в набор элементов орбиты
    a, e, i, Omega, omega, M

    """
    r = state[:3]
    v = state[3:]
    r_abs = np.linalg.norm(r)
    v_abs = np.linalg.norm(v)

    c = np.cross(r, v)                   # Вектор орбительного момента
    fl = np.cross(v, c) - mu * r / r_abs # Вектор Лапласа

    f_abs = np.linalg.norm(fl)  # Величина fl
    c_abs = np.linalg.norm(c)   # Величина c

    # Параметр орбиты и эксцентриситет
    p = c_abs * c_abs / mu
    e = f_abs / mu
    a = p / (1. - e * e)        # Большая полуось
    
    i = np.arccos(c[2] / c_abs)     # Наклонение, рад

    node = np.cross(np.array([0., 0., 1.]), c) # Вектор в линии узлов
    node = node / np.linalg.norm(node)

    Omega = np.arctan2(node[1], node[0]) # ДВУ по ориентации вектора узла

    omega = np.arccos(np.dot(fl, node) / f_abs) # cos из ориентации узла и вектора Лапласа
    if fl[2] < 0.0 : omega = 2. * np.pi - omega # корректируем квадрант

    nu = np.arccos(np.dot(fl, r) / f_abs / r_abs) # угол между вектором Лапласа и вектором положения
    if (np.dot(r, v) < 0): nu = 2. * np.pi - nu

    E = 2. * np.arctan(np.tan(nu * 0.5) * np.sqrt((1. - e) / (1. + e)))

    # cosE = (e + np.cos(nu)) / (1. + e * np.cos(nu))
    # E = np.arccos(cosE)
    # if nu > np.pi and (nu < 2. * np.pi): E = 2. * np.pi - E
    M = E - e * np.sin(E)

    return np.array([a, e, i, Omega, omega, M])


def xyz2llh(r: np.ndarray, Re: float = 6.3781365,
            F: float = 0.00335286, tol: float = 5.e-8):
    """ Переводит прямоугольные координаты в широту, долготу и высоту

    Args:
       r:  Вектор или список из трех коррдиант положения точки, тыс. км
       Re: Экваториальный радиус Земли, тыс. км
       F:  Параметр сжатия Земли, б/р.

    Returns:
      Кортеж из трех значений:
      - геодезическая широта, радианы
      - геодезическая долгота, радианы
      - высота над порвехностью эллипсоида, тыс. км
    
    """
    x, y, z = r
    e = np.sqrt(2 * F - F * F)
    dz = 0.0

    i = 0
    xy2 = x * x + y * y
    phi0 = -10
    N = 0
    
    while i < 10:
        sin_phi = (z + dz) / np.sqrt(xy2 + (z + dz) * (z + dz))
        N = Re / np.sqrt(1 - e * e * sin_phi * sin_phi)
        dz = N * e * e * sin_phi

        phi = np.arcsin(sin_phi)
        if abs(phi - phi0) < tol:
            break

        phi0 = phi
        i += 1

    phi = np.arctan((z + dz) / np.sqrt(xy2))
    lam = np.arctan2(y, x)
    h = np.sqrt(xy2 + (z + dz) * (z + dz)) - N

    return phi, lam, h


def rotate_to_earth(t: datetime):
    t0 = datetime.fromisoformat('2025-02-03 03:07:04')
    dt = (t - t0).total_seconds()
    theta = omega0 * dt * 1.e-3 # Время должно быть в тыс. секунд
    A = np.eye(3)
    A[0, 0] = np.cos(theta)
    A[1, 1] = A[0, 0]
    A[0, 1] = np.sin(theta)
    A[1, 0] = -A[0, 1]
    return A


def pq_from_elems(i: float, Omega: float, omega: float):
    """Получение значений базисных векторов P и Q из элементов орбиты.

    """
    P = np.array([
        np.cos(omega) * np.cos(Omega) - np.sin(omega) * np.cos(i) * np.sin(Omega),
        np.cos(omega) * np.sin(Omega) + np.sin(omega) * np.cos(i) * np.cos(Omega),
        np.sin(omega) * np.sin(i)
    ])
    Q = np.array([
        -np.sin(omega) * np.cos(Omega) - np.cos(omega) * np.cos(i) * np.sin(Omega),
        -np.sin(omega) * np.sin(Omega) + np.cos(omega) * np.cos(i) * np.cos(Omega),
        +np.cos(omega) * np.sin(i)
    ])

    return P, Q

