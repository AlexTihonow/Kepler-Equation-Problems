import numpy as np
from datetime import datetime, timedelta
from extra import (
    muE, sigma, tau, omega0,
    calc_rho_atm, rk4, solve_kepler,
    elems2state, state2elems,
    xyz2llh, rotate_to_earth, pq_from_elems
)

def rhs_elements(t: float, alpha: np.ndarray) -> np.ndarray:
    """Правые части уравнений Гаусса для оскулирующих элементов.
    t в тыс. сек, alpha = (a, e, i, Ω, ω, M), a в тыс. км.
    """
    a, e, i, Omega, omega, M = alpha

    p = a * (1. - e * e)
    n = np.sqrt(muE / a**3)

    E  = solve_kepler(e, M)
    nu = 2. * np.arctan(np.tan(E * 0.5) * np.sqrt((1. + e) / (1. - e)))
    u  = nu + omega
    r  = p / (1. + e * np.cos(nu))

    # Базисные векторы орбитальной СК
    P, Q = pq_from_elems(i, Omega, omega)

    # Вектор положения в инерциальной СК (тыс. км)
    r_vec = P * r * np.cos(nu) + Q * r * np.sin(nu)

    # Высота над эллипсоидом: xyz2llh возвращает h в тыс. км
    _, _, h_tkm = xyz2llh(r_vec)

    # Плотность атмосферы кг/м³ (calc_rho_atm принимает h в тыс. км)
    rho = calc_rho_atm(h_tkm)

    # Скорость в орбитальной СК (км/с = тыс.км/тыс.сек)
    v_r  = np.sqrt(muE / p) * e * np.sin(nu)
    v_nu = np.sqrt(muE / p) * (1. + e * np.cos(nu))

    # Относительная скорость (атмосфера вращается с Землёй)
    v_rel = np.array([
        v_r,
        v_nu - omega0 * r * np.cos(i),
        omega0 * r * np.sin(i) * np.cos(u)
    ])
    v_rel_norm = np.linalg.norm(v_rel)  # км/с

    # Возмущающее ускорение атмосферы
    # sigma=1 м²/кг, rho кг/м³, v в км/с → перевод в тыс.км/тыс.сек²:
    #   a[м/с²] * 1e-3 → км/с² = тыс.км/тыс.сек²


    # ПРАВИЛЬНО: СИ → тыс.км/тыс.сек²
    a_atm = -sigma * rho * (v_rel_norm * 1e3) * (v_rel * 1e3)# тыс.км/тыс.сек²
    S, T, W = a_atm

    sq  = np.sqrt(1. - e * e)
    spp = np.sqrt(p / muE)
    si  = np.sin(i) if abs(np.sin(i)) > 1e-10 else 1e-10

    da  = 2. / (n * sq) * (e * S * np.sin(nu) + T * p / r)
    de  = spp * (S * np.sin(nu) + T * ((1. + r/p) * np.cos(nu) + r*e/p))
    di  = r * W * np.cos(u) / np.sqrt(muE * p)
    dOm = r * W * np.sin(u) / (np.sqrt(muE * p) * si)
    dom = (1./e) * spp * (T * (1.+r/p) * np.sin(nu) - S * np.cos(nu)) \
          - r * W * np.sin(u) * np.cos(i) / (np.sqrt(muE * p) * si)
    dM  = n \
          - 2. * S * r / np.sqrt(muE * a) \
          + sq / e * spp * (S * np.cos(nu) - T * (1.+r/p) * np.sin(nu))

    return np.array([da, de, di, dOm, dom, dM])


def propagate(r0_tkm: np.ndarray, v0_kms: np.ndarray, t0: datetime) -> dict:
    """
    Главный алгоритм (п.3 лабораторной).
    r0_tkm : начальный вектор положения, тыс. км
    v0_kms : начальный вектор скорости, км/с (= тыс.км/тыс.сек)
    t0     : момент начальных условий (datetime UTC)
    """
    # Шаг 1: X9 → α0
    state0 = np.concatenate([r0_tkm, v0_kms])
    alpha  = state2elems(state0)
    t      = 0.0    # тыс. сек от t0
    t_cur  = t0

    h_drop = 0.150   # 150 км = 0.150 тыс. км
    t_max  = 7776.0  # 3 месяца в тыс. сек
    fell   = False

    # Шаг 2: τ = tau = 0.060 тыс. сек (из extra.py)
    # Шаги 3–7: основной цикл
    while True:
        # Шаг 4: RK4 → α_{i+1}
        alpha_new = rk4(rhs_elements, t, alpha, tau)
        t_new     = t + tau
        t_cur_new = t0 + timedelta(seconds=t_new * 1e3)

        # Шаг 5: α_{i+1} → r_{i+1}
        state_new = elems2state(alpha_new)
        
        r_new     = state_new[:3]

        # Шаг 6: высота над эллипсоидом
        _, _, h_tkm = xyz2llh(r_new)
        

        alpha = alpha_new
        t     = t_new
        t_cur = t_cur_new

        # Шаг 7: условия остановки
        if h_tkm < h_drop:
            fell = True
            break
        if t > t_max:
            fell = False
            break

    # Шаг 8: поворот в гринвичскую СК через rotate_to_earth
    A    = rotate_to_earth(t_cur)
    r_gr = A @ r_new

    # Шаг 9: геодезические координаты
    phi, lam, h_fin = xyz2llh(r_gr)

    return {
        "fell":    fell,
        "t_tsec":  t,
        "t_end":   t_cur,
        "lat_deg": np.degrees(phi),
        "lon_deg": np.degrees(lam),
        "h_km":    h_fin * 1e3,
    }



#| 20 | Тихонов Александр Романович     |  26513 | 2026-02-18 00:21:57.586 | -6.317170 | -3.616757 |  0.016259 |  1.582410 | -2.705613 | 6.719165 |
# ===== Начальные условия из п.4 (МКС) =====
# |  2.972117 |  6.049850 | -0.007805 | -5.041246 |  2.462888 | 5.259396 |

r0 = np.array([-6.317170,   -3.616757,   0.016259])  # тыс. км
v0 = np.array([1.582410,  -2.705613,  6.719165   ])  # км/с

t0_dt = datetime(2026, 2, 18, 0, 21, 57, 586000)

result = propagate(r0, v0, t0_dt)

print(f"Падение:         {result['fell']}")
print(f"Время от t0:     {result['t_tsec']:.3f} тыс. сек")
print(f"Конечный момент: {result['t_end']}")
print(f"Широта:          {result['lat_deg']:.3f}°")
print(f"Долгота:         {result['lon_deg']:.3f}°")
print(f"Высота:          {result['h_km']:.3f} км")
