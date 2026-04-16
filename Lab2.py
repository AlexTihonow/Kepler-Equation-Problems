import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.integrate import solve_ivp
from datetime import datetime, timedelta
from sgp4.api import Satrec, WGS84, jday

# Импорт функции из вашего файла harm.py
from harm import calc_geop_accel

# ==================== ПАРАМЕТРЫ ЗАДАЧИ ====================
Re_km = 6378.1365
muE_km = 398600.4415
sigma = 0.01                 # Баллистический коэффициент [м2/кг]
omega_earth = 7.292115085e-5 # Скорость вращения Земли [рад/с]
J2 = 0.108262668355315e-02
t_earth_epoch = datetime(2025, 2, 3, 3, 7, 4)

# Начальные условия МКС (x0 задан в тысячах км, переводим в км)
x0 = np.array([-2.17787461e+00, -6.43542127e+00, 5.55924609e-03, 
                4.51953616e+00, -1.51712974e+00, 5.99827094e+00])
r0_km = x0[:3] * 1000.0  
v0_km = x0[3:]           
t0 = datetime(2025, 2, 3, 0, 7, 59, 378000)

tle_line1 = '1 25544U 98067A   25033.96763762  .00018191  00000-0  32891-3 0  9995'
tle_line2 = '2 25544  51.6374 251.8191 0003396 268.5169 232.7283 15.49742755494343'

# ================= ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =================
def calc_altitude(r_vec):
    """Итерационный метод расчета высоты над общеземным эллипсоидом."""
    F = 1 / 298.2525784
    e2 = 2 * F - F**2
    x, y, z = r_vec
    r_xy = np.sqrt(x**2 + y**2)
    dz = 0.0
    phi_prev = -1000
    for _ in range(10):
        sin_phi = (z + dz) / np.sqrt(r_xy**2 + (z + dz)**2)
        N = Re_km / np.sqrt(1 - e2 * sin_phi**2)
        dz = N * e2 * sin_phi
        phi = np.arctan2((z + dz), r_xy)
        if abs(phi - phi_prev) < 1e-8:
            break
        phi_prev = phi
    return np.sqrt(r_xy**2 + (z + dz)**2) - N

def get_density(h):
    """Табличная модель атмосферы (Приложение А). Возвращает кг/м3."""
    if h < 150: return 2.070e-9 * np.exp(-(h - 150) / 22.523)
    elif h < 180: return 2.070e-9 * np.exp(-(h - 150) / 22.523)
    elif h < 200: return 5.464e-10 * np.exp(-(h - 180) / 29.740)
    elif h < 250: return 2.789e-10 * np.exp(-(h - 200) / 37.105)
    elif h < 300: return 7.248e-11 * np.exp(-(h - 250) / 45.546)
    elif h < 350: return 2.418e-11 * np.exp(-(h - 300) / 53.628)
    elif h < 400: return 9.518e-12 * np.exp(-(h - 350) / 53.298)
    elif h < 450: return 3.725e-12 * np.exp(-(h - 400) / 58.515)
    elif h < 500: return 1.585e-12 * np.exp(-(h - 450) / 60.828)
    elif h < 600: return 6.967e-13 * np.exp(-(h - 500) / 63.822)
    elif h < 700: return 1.454e-13 * np.exp(-(h - 600) / 71.835)
    elif h < 800: return 3.614e-14 * np.exp(-(h - 700) / 88.667)
    elif h < 900: return 1.170e-14 * np.exp(-(h - 800) / 124.64)
    elif h < 1000: return 5.245e-15 * np.exp(-(h - 900) / 181.05)
    elif h < 1500: return 3.019e-15 * np.exp(-(h - 1000) / 268.00)
    return 0.0

def rv2elements(r_vec, v_vec):
    """Перевод r, v в оскулирующие элементы a, e, Omega, omega, i."""
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)
    
    eps = v**2 / 2 - muE_km / r
    a = -muE_km / (2 * eps)
    
    e_vec = np.cross(v_vec, h_vec) / muE_km - r_vec / r
    e = np.linalg.norm(e_vec)
    
    i = np.arccos(h_vec[2] / h)
    n_vec = np.cross([0, 0, 1], h_vec)
    n = np.linalg.norm(n_vec)
    
    if n != 0:
        Omega = np.arccos(n_vec[0] / n)
        if n_vec[1] < 0:
            Omega = 2 * np.pi - Omega
        
        omega = np.arccos(np.clip(np.dot(n_vec, e_vec) / (n * e), -1.0, 1.0))
        if e_vec[2] < 0:
            omega = 2 * np.pi - omega
    else:
        Omega = 0; omega = 0
        
    return a, e, np.degrees(Omega), np.degrees(omega), i

# ==================== ЧИСЛЕННАЯ МОДЕЛЬ ====================
def numerical_model(t, y):
    r_vec = y[:3]
    v_vec = y[3:]
    r_norm = np.linalg.norm(r_vec)
    
    # Матрица поворота Земли
    current_time = t0 + timedelta(seconds=float(t))
    dt_earth = (current_time - t_earth_epoch).total_seconds()
    theta = omega_earth * dt_earth
    A_t = np.array([
        [np.cos(theta),  np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0,              0,             1]
    ])
    
    # Гравитация нецентрального поля
    r_earth_Mm = (A_t @ r_vec) / 1000.0
    f_geop_earth_ms2 = calc_geop_accel(r_earth_Mm)
    f_geop_kms2 = (A_t.T @ f_geop_earth_ms2) / 1000.0
    
    # Торможение атмосферы
    v_rel = v_vec - np.cross([0, 0, omega_earth], r_vec)
    v_rel_norm = np.linalg.norm(v_rel)
    h = calc_altitude(r_vec)
    rho = get_density(h)
    
    # Правильный множитель 1000.0
    f_atm_kms2 = -1000.0 * sigma * rho * v_rel_norm * v_rel
    
    a_vec = -muE_km * r_vec / r_norm**3 + f_geop_kms2 + f_atm_kms2
    return np.concatenate((v_vec, a_vec))

def event_150km(t, y):
    """Остановка интегрирования, если высота падает ниже 150 км"""
    return calc_altitude(y[:3]) - 150.0
event_150km.terminal = True

# ==================== ОСНОВНОЙ РАСЧЕТ ====================
max_time = 30 * 86400  # 30 дней
step = 600             # 10 минут
eval_times = np.arange(0, max_time + step, step)

print("Запуск численной модели (это может занять около 10-30 секунд)...")
sol = solve_ivp(numerical_model, [0, max_time], np.concatenate((r0_km, v0_km)), 
                t_eval=eval_times, events=event_150km, method='RK45', rtol=1e-8, atol=1e-8)
times = sol.t

results_num = []
results_sgp = []
results_kp  = []

# Инициализация для SGP4
sat = Satrec.twoline2rv(tle_line1, tle_line2, WGS84)

# Инициализация для Кеплер+
a0, e0, Omega0_deg, omega0_deg, i0 = rv2elements(r0_km, v0_km)
Omega0, omega0 = np.radians(Omega0_deg), np.radians(omega0_deg)
n0 = np.sqrt(muE_km / a0**3)
p0 = a0 * (1 - e0**2)
ndot_TLE = 0.00018191  # Из строки TLE
ndot = 2 * ndot_TLE * 2 * np.pi / (86400**2) # В рад/с2

print("Синхронный расчет для SGP4 и Кеплер+ ...")
for i, t_sec in enumerate(times):
    # 1. Численная
    r_num = sol.y[:3, i]
    v_num = sol.y[3:, i]
    results_num.append(rv2elements(r_num, v_num)[:4])
    
    # 2. SGP4
    current_dt = t0 + timedelta(seconds=float(t_sec))
    jd, fr = jday(current_dt.year, current_dt.month, current_dt.day, 
                  current_dt.hour, current_dt.minute, current_dt.second + current_dt.microsecond * 1e-6)
    e_code, r_sgp, v_sgp = sat.sgp4(jd, fr)
    results_sgp.append(rv2elements(np.array(r_sgp), np.array(v_sgp))[:4])
    
    # 3. Кеплер+
    a_kp = a0 - (2 * a0 / (3 * n0)) * ndot * t_sec
    e_kp = e0 - (2 * (1 - e0) / (3 * n0)) * ndot * t_sec
    Omega_kp = Omega0 - (3 * n0 * Re_km**2 * J2 / (2 * p0**2)) * np.cos(i0) * t_sec
    omega_kp = omega0 + (3 * n0 * Re_km**2 * J2 / (4 * p0**2)) * (4 - 5 * np.sin(i0)**2) * t_sec
    
    results_kp.append((a_kp, e_kp, np.degrees(Omega_kp % (2*np.pi)), np.degrees(omega_kp % (2*np.pi))))

# ==================== ГРАФИКИ ====================
results_num = np.array(results_num)
results_sgp = np.array(results_sgp)
results_kp  = np.array(results_kp)

dates = [t0 + timedelta(seconds=float(t)) for t in times]

plt.style.use('ggplot')
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

labels = ['tle', 'num', 'kepler+']
colors = ['#348ABD', '#A60628', '#7A68A6']

# [0, 0] Большая полуось (в тыс. км)
axs[0, 0].plot(dates, results_sgp[:, 0] / 1000.0, label=labels[0], color=colors[0])
axs[0, 0].plot(dates, results_num[:, 0] / 1000.0, label=labels[1], color=colors[1])
axs[0, 0].plot(dates, results_kp[:, 0] / 1000.0,  label=labels[2], color=colors[2])
axs[0, 0].set_title('Большая полуось, a')
axs[0, 0].set_ylabel('тыс. км')

# [0, 1] Эксцентриситет
axs[0, 1].plot(dates, results_sgp[:, 1], label=labels[0], color=colors[0])
axs[0, 1].plot(dates, results_num[:, 1], label=labels[1], color=colors[1])
axs[0, 1].plot(dates, results_kp[:, 1],  label=labels[2], color=colors[2])
axs[0, 1].set_title('Эксцентриситет, e')
axs[0, 1].set_ylabel('б/р')

# Функции для приведения углов
def wrap_to_pi(deg_arr):
    rad = np.radians(deg_arr)
    return (rad + np.pi) % (2 * np.pi) - np.pi

def wrap_to_2pi(deg_arr):
    return np.radians(deg_arr) % (2 * np.pi)

# [1, 0] ДВУ (в радианах, от -pi до pi)
axs[1, 0].plot(dates, wrap_to_pi(results_sgp[:, 2]), label=labels[0], color=colors[0])
axs[1, 0].plot(dates, wrap_to_pi(results_num[:, 2]), label=labels[1], color=colors[1])
axs[1, 0].plot(dates, wrap_to_pi(results_kp[:, 2]),  label=labels[2], color=colors[2])
axs[1, 0].set_title('ДВУ, $\Omega$')
axs[1, 0].set_ylabel('рад')

# [1, 1] Арг. перицентра (в радианах, от 0 до 2pi)
axs[1, 1].plot(dates, wrap_to_2pi(results_sgp[:, 3]), label=labels[0], color=colors[0])
axs[1, 1].plot(dates, wrap_to_2pi(results_num[:, 3]), label=labels[1], color=colors[1])
axs[1, 1].plot(dates, wrap_to_2pi(results_kp[:, 3]),  label=labels[2], color=colors[2])
axs[1, 1].set_title('Арг. перицентра, $\omega$')
axs[1, 1].set_ylabel('рад')

for ax in axs.flat:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.tick_params(axis='x', rotation=30)
    if ax in [axs[1, 0], axs[1, 1]]:
        ax.set_xlabel('Дата/время')
    ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig('lab_02_results_styled.png', dpi=300)
print("Готово! Графики сохранены в файл 'lab_02_results_styled.png'")
plt.show()
