import numpy as np
from scipy.linalg import inv
import matplotlib.pyplot as plt
from typing import Callable, Tuple

class BornTeplyAdaptiveKalmanFilter:
    """
    Реализация обобщенного алгоритма Борна-Тейпли
    для совместной оценки состояния и матрицы R
    """
    
    def __init__(self, F, H, Q, R_init, x0, P0, B=None):
        """
        Инициализация фильтра
        
        Parameters:
        -----------
        F : np.ndarray or callable
            Матрица перехода состояния F_k
        H : np.ndarray or callable  
            Матрица измерений H_k
        Q : np.ndarray or callable
            Ковариация шума процесса Q_k
        R_init : np.ndarray
            Начальная оценка матрицы R
        x0 : np.ndarray
            Начальное состояние
        P0 : np.ndarray
            Начальная ковариация ошибки
        B : np.ndarray or callable, optional
            Матрица B_k, по умолчанию единичная
        """
        self.F = F if callable(F) else lambda k: F
        self.H = H if callable(H) else lambda k: H
        self.Q = Q if callable(Q) else lambda k: Q
        self.B = B if callable(B) else (lambda k: B if B is not None else np.eye(R_init.shape[0]))
        
        # Инициализация состояния и ковариаций
        self.x = x0.copy()
        self.P = P0.copy()
        self.R = R_init.copy()
        
        # Начальные условия
        self.n_state = x0.shape[0]
        self.n_meas = R_init.shape[0]
        self.k = 1  # Счетчик шагов для адаптации R
        
        # История для анализа
        self.history = {
            'x_est': [], 'P_diag': [], 'R_est': [], 
            'v': [], 'K_norm': [], 'time': []
        }
    
    def predict(self, step: int) -> np.ndarray:
        """
        Шаг прогноза (уравнения 5-6)
        """
        F_k = self.F(step)
        Q_k = self.Q(step)
        
        # Прогноз состояния
        self.x_pred = F_k @ self.x
        # Прогноз ковариации
        self.P_pred = F_k @ self.P @ F_k.T + Q_k
        
        return self.x_pred
    
    def update(self, step: int, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Шаг коррекции с адаптацией R (уравнения 7-12)
        """
        H_k = self.H(step)
        B_k = self.B(step)
        
        # Невязка (уравнение 9)
        v = measurement - H_k @ self.x_pred
        
        # Ковариация невязки (уравнение 10)
        C = H_k @ self.P_pred @ H_k.T + B_k @ self.R @ B_k.T
        
        # Коэффициент Калмана (уравнение 8)
        K = self.P_pred @ H_k.T @ inv(C)
        
        # Коррекция состояния (уравнение 7)
        self.x = self.x_pred + K @ v
        
        # Коррекция ковариации (уравнение 11)
        I = np.eye(self.n_state)
        self.P = (I - K @ H_k) @ self.P_pred
        
        # АДАПТАЦИЯ R (уравнение 12)
        self._adapt_R(step, v, B_k)
        
        # Сохранение истории
        self._save_history(v, K)
        
        return self.x.copy(), self.R.copy()
    
    def _adapt_R(self, step: int, v: np.ndarray, B: np.ndarray):
        """
        Адаптация матрицы R по уравнению (12)
        
        R_k = (1/k)*(B^T B)^{-1} B^T (v v^T) B (B^T B)^{-1} + (1-1/k)*R_{k-1}
        """
        # Вычисление (B^T B)^{-1}
        BtB = B.T @ B
        BtB_inv = inv(BtB)
        
        # Внешнее произведение невязки
        v_outer = np.outer(v, v)
        
        # Первое слагаемое
        first_term = BtB_inv @ B.T @ v_outer @ B @ BtB_inv
        
        # Весовые коэффициенты
        weight = 1.0 / self.k
        
        # Рекуррентное обновление R
        self.R = weight * first_term + (1 - weight) * self.R
        
        # Увеличение счетчика
        self.k += 1
    
    def _save_history(self, v: np.ndarray, K: np.ndarray):
        """Сохранение истории оценок"""
        self.history['x_est'].append(self.x.copy())
        self.history['P_diag'].append(np.diag(self.P).copy())
        self.history['R_est'].append(self.R.copy())
        self.history['v'].append(v.copy())
        self.history['K_norm'].append(np.linalg.norm(K))
        self.history['time'].append(len(self.history['x_est']))

class ClassicalKalmanFilter:
    """
    Классический фильтр Калмана для сравнения
    """
    def __init__(self, F, H, Q, R, x0, P0):
        self.F = F if callable(F) else lambda k: F
        self.H = H if callable(H) else lambda k: H
        self.Q = Q if callable(Q) else lambda k: Q
        self.R = R if callable(R) else lambda k: R
        self.x = x0.copy()
        self.P = P0.copy()
        self.history = {'x_est': []}
    
    def predict(self, step: int):
        F_k = self.F(step)
        Q_k = self.Q(step)
        self.x = F_k @ self.x
        self.P = F_k @ self.P @ F_k.T + Q_k
    
    def update(self, step: int, measurement: np.ndarray):
        H_k = self.H(step)
        R_k = self.R(step)
        
        # Невязка
        v = measurement - H_k @ self.x
        
        # Коэффициент Калмана
        S = H_k @ self.P @ H_k.T + R_k
        K = self.P @ H_k.T @ inv(S)
        
        # Коррекция
        self.x = self.x + K @ v
        I = np.eye(self.x.shape[0])
        self.P = (I - K @ H_k) @ self.P
        
        self.history['x_est'].append(self.x.copy())
        return self.x

import numpy as np
import pandas as pd
from scipy.linalg import inv
import matplotlib.pyplot as plt

def calculate_armse(true_states, estimated_states):
    """
    Вычисление ARMSE (Average Root Mean Square Error)
    """
    true_states = np.array(true_states)
    estimated_states = np.array(estimated_states)
    
    # Разница между истинным и оцененным
    errors = true_states - estimated_states
    
    # RMSE для каждого компонента состояния
    rmse_per_component = np.sqrt(np.mean(errors**2, axis=0))
    
    # ARMSE - среднее по всем компонентам
    armse = np.mean(rmse_per_component)
    
    return armse

def run_grid_experiment(n_steps=100, n_state=2, seed=42):
    """
    Запуск сеточного эксперимента для сравнения AKF и KF
    с разными масштабами Q и R
    
    Parameters:
    -----------
    n_steps : int, количество шагов моделирования
    n_state : int, размерность состояния
    seed : int, seed для воспроизводимости
    
    Returns:
    --------
    dict: Словарь с результатами ARMSE для всех комбинаций
    """
    np.random.seed(seed)
    
    # Базовые матрицы
    F_base = np.array([[1, 0.1], [0, 1]])
    H_base = np.eye(2)
    Q_base = np.eye(2) * 0.01
    R_base = np.eye(2) * 0.25
    
    # Начальные условия
    x0 = np.array([0.0, 0.0])
    P0 = np.eye(2) * 1.0
    
    # Масштабные коэффициенты
    q_scales = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    r_scales = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    
    # Результаты
    results_akf = np.zeros((len(r_scales), len(q_scales)))
    results_kf = np.zeros((len(r_scales), len(q_scales)))
    
    # Генерация тестовых данных
    true_states = []
    x_true = np.array([0.0, 1.0])
    for k in range(n_steps):
        if k > 0:
            x_true = np.array([x_true[0] + x_true[1]*0.1, x_true[1]])
        true_states.append(x_true.copy())
    
    # Генерация зашумленных измерений (фиксированные для всех экспериментов)
    measurements = []
    for state in true_states:
        noise = np.random.normal(0, 0.5, size=n_state)
        measurements.append(state + noise)
    
    # Прогон по всем комбинациям
    for i, r_scale in enumerate(r_scales):
        for j, q_scale in enumerate(q_scales):
            # Масштабирование матриц
            Q = Q_base * q_scale
            R_true = R_base * r_scale
            R_init = R_base * r_scale  # Такая же инициализация для AKF
            
            # Создание фильтров
            akf = BornTeplyAdaptiveKalmanFilter(F_base, H_base, Q, R_init, x0, P0)
            kf = ClassicalKalmanFilter(F_base, H_base, Q, R_true, x0, P0)
            
            # Запуск фильтров
            akf_estimates = []
            kf_estimates = []
            
            for k in range(n_steps):
                # Адаптивный фильтр
                akf.predict(k)
                x_akf, _ = akf.update(k, measurements[k])
                akf_estimates.append(x_akf)
                
                # Классический фильтр
                kf.predict(k)
                x_kf = kf.update(k, measurements[k])
                kf_estimates.append(x_kf)
            
            # Вычисление ARMSE
            results_akf[i, j] = calculate_armse(true_states, akf_estimates)
            results_kf[i, j] = calculate_armse(true_states, kf_estimates)
            
            print(f"Прогресс: R_scale={r_scale:.3f}, Q_scale={q_scale:.3f} - "
                  f"AKF: {results_akf[i, j]:.6f}, KF: {results_kf[i, j]:.6f}")
    
    return {
        'akf': results_akf,
        'kf': results_kf,
        'q_scales': q_scales,
        'r_scales': r_scales
    }

def create_tables(results):
    """
    Создание таблиц в формате как в примере
    
    Parameters:
    -----------
    results : dict, результаты из run_grid_experiment
    """
    akf_results = results['akf']
    kf_results = results['kf']
    q_scales = results['q_scales']
    r_scales = results['r_scales']
    
    # 1. Таблица ARMSE AKF
    print("\n" + "="*80)
    print("ARMSE AKF")
    print("="*80)
    
    # Заголовки
    header = "| ARMSE    | "
    for q in q_scales:
        if q == 0.001:
            header += "0.001Q | "
        elif q == 0.01:
            header += "0.011Q | "
        elif q == 0.1:
            header += "0.11Q | "
        elif q == 1:
            header += "Q    | "
        elif q == 10:
            header += "10°Q | "
        elif q == 100:
            header += "100°Q | "
        elif q == 1000:
            header += "1000°Q | "
    
    print(header)
    print("|" + "-"*10 + "|" + "|".join(["-"*8 for _ in q_scales]) + "|")
    
    # Данные
    for i, r in enumerate(r_scales):
        row = "| "
        if r == 0.001:
            row += "0.001*R   | "
        elif r == 0.01:
            row += "0.001*R   | "  # В примере тоже 0.001*R
        elif r == 0.1:
            row += "0.001*R   | "  # В примере тоже 0.001*R
        elif r == 1:
            row += "R    | "
        elif r == 10:
            row += "10°R    | "
        elif r == 100:
            row += "100°R   | "
        elif r == 1000:
            row += "1000°R  | "
        
        for j in range(len(q_scales)):
            row += f"{akf_results[i, j]:.6f} | "
        print(row)
    
    # 2. Таблица ARMSE KF
    print("\n" + "="*80)
    print("ARMSE KF")
    print("="*80)
    
    print(header)
    print("|" + "-"*10 + "|" + "|".join(["-"*8 for _ in q_scales]) + "|")
    
    for i, r in enumerate(r_scales):
        row = "| "
        if r == 0.001:
            row += "0.001*R   | "
        elif r == 0.01:
            row += "0.001*R   | "
        elif r == 0.1:
            row += "0.001*R   | "
        elif r == 1:
            row += "R    | "
        elif r == 10:
            row += "10°R    | "
        elif r == 100:
            row += "100°R   | "
        elif r == 1000:
            row += "1000°R  | "
        
        for j in range(len(q_scales)):
            row += f"{kf_results[i, j]:.6f} | "
        print(row)
    
    # 3. Таблица разницы
    print("\n" + "="*80)
    print("Разница между ARMSE AKF и ARMSE KF")
    print("="*80)
    
    difference = kf_results - akf_results
    
    print(header)
    print("|" + "-"*10 + "|" + "|".join(["-"*8 for _ in q_scales]) + "|")
    
    for i, r in enumerate(r_scales):
        row = "| "
        if r == 0.001:
            row += "0.001*R   | "
        elif r == 0.01:
            row += "0.001*R   | "
        elif r == 0.1:
            row += "0.001*R   | "
        elif r == 1:
            row += "R    | "
        elif r == 10:
            row += "10°R    | "
        elif r == 100:
            row += "100°R   | "
        elif r == 1000:
            row += "1000°R  | "
        
        for j in range(len(q_scales)):
            row += f"{difference[i, j]:.6f} | "
        print(row)
    
    return {
        'akf_table': akf_results,
        'kf_table': kf_results,
        'diff_table': difference
    }

def save_tables_to_excel(results, filename="kalman_comparison.xlsx"):
    """
    Сохранение таблиц в Excel файл
    
    Parameters:
    -----------
    results : dict, результаты из run_grid_experiment
    filename : str, имя файла для сохранения
    """
    akf_results = results['akf']
    kf_results = results['kf']
    q_scales = results['q_scales']
    r_scales = results['r_scales']
    difference = kf_results - akf_results
    
    # Создание DataFrame для каждой таблицы
    # Форматирование заголовков как в примере
    q_labels = []
    for q in q_scales:
        if q == 0.001:
            q_labels.append("0.001Q")
        elif q == 0.01:
            q_labels.append("0.011Q")
        elif q == 0.1:
            q_labels.append("0.11Q")
        elif q == 1:
            q_labels.append("Q")
        elif q == 10:
            q_labels.append("10°Q")
        elif q == 100:
            q_labels.append("100°Q")
        elif q == 1000:
            q_labels.append("1000°Q")
    
    r_labels = []
    for r in r_scales:
        if r == 0.001:
            r_labels.append("0.001*R")
        elif r == 0.01:
            r_labels.append("0.001*R")
        elif r == 0.1:
            r_labels.append("0.001*R")
        elif r == 1:
            r_labels.append("R")
        elif r == 10:
            r_labels.append("10°R")
        elif r == 100:
            r_labels.append("100°R")
        elif r == 1000:
            r_labels.append("1000°R")
    
    # Создание DataFrame
    df_akf = pd.DataFrame(akf_results, index=r_labels, columns=q_labels)
    df_kf = pd.DataFrame(kf_results, index=r_labels, columns=q_labels)
    df_diff = pd.DataFrame(difference, index=r_labels, columns=q_labels)
    
    # Сохранение в Excel
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        df_akf.to_excel(writer, sheet_name='ARMSE_AKF')
        df_kf.to_excel(writer, sheet_name='ARMSE_KF')
        df_diff.to_excel(writer, sheet_name='Разница_ARMSE')
        
        # Добавление форматирования
        workbook = writer.book
        worksheet_akf = writer.sheets['ARMSE_AKF']
        worksheet_kf = writer.sheets['ARMSE_KF']
        worksheet_diff = writer.sheets['Разница_ARMSE']
        
        # Установка ширины колонок
        for worksheet in [worksheet_akf, worksheet_kf, worksheet_diff]:
            worksheet.column_dimensions['A'].width = 12
            for col in range(1, len(q_labels) + 1):
                worksheet.column_dimensions[chr(65 + col)].width = 10
    
    print(f"\nТаблицы сохранены в файл: {filename}")
    return df_akf, df_kf, df_diff

def visualize_results(results):
    """
    Визуализация результатов сеточного эксперимента
    
    Parameters:
    -----------
    results : dict, результаты из run_grid_experiment
    """
    akf_results = results['akf']
    kf_results = results['kf']
    q_scales = results['q_scales']
    r_scales = results['r_scales']
    difference = kf_results - akf_results
    
    # Создание подграфиков
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. ARMSE AKF
    im1 = axes[0].imshow(akf_results, cmap='viridis', aspect='auto')
    axes[0].set_title('ARMSE AKF', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Масштаб Q', fontsize=12)
    axes[0].set_ylabel('Масштаб R', fontsize=12)
    
    # Настройка осей
    q_labels = [f"{q:.3f}" if q < 1 else f"{q:.0f}" for q in q_scales]
    r_labels = [f"{r:.3f}" if r < 1 else f"{r:.0f}" for r in r_scales]
    
    axes[0].set_xticks(range(len(q_scales)))
    axes[0].set_xticklabels(q_labels, rotation=45)
    axes[0].set_yticks(range(len(r_scales)))
    axes[0].set_yticklabels(r_labels)
    
    # Добавление значений в ячейки
    for i in range(len(r_scales)):
        for j in range(len(q_scales)):
            axes[0].text(j, i, f'{akf_results[i, j]:.3f}', 
                       ha='center', va='center', color='w', fontsize=8)
    
    # 2. ARMSE KF
    im2 = axes[1].imshow(kf_results, cmap='plasma', aspect='auto')
    axes[1].set_title('ARMSE KF', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Масштаб Q', fontsize=12)
    axes[1].set_ylabel('Масштаб R', fontsize=12)
    axes[1].set_xticks(range(len(q_scales)))
    axes[1].set_xticklabels(q_labels, rotation=45)
    axes[1].set_yticks(range(len(r_scales)))
    axes[1].set_yticklabels(r_labels)
    
    for i in range(len(r_scales)):
        for j in range(len(q_scales)):
            axes[1].text(j, i, f'{kf_results[i, j]:.3f}', 
                       ha='center', va='center', color='w', fontsize=8)
    
    # 3. Разница
    im3 = axes[2].imshow(difference, cmap='RdYlBu', aspect='auto')
    axes[2].set_title('Разница (KF - AKF)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Масштаб Q', fontsize=12)
    axes[2].set_ylabel('Масштаб R', fontsize=12)
    axes[2].set_xticks(range(len(q_scales)))
    axes[2].set_xticklabels(q_labels, rotation=45)
    axes[2].set_yticks(range(len(r_scales)))
    axes[2].set_yticklabels(r_labels)
    
    for i in range(len(r_scales)):
        for j in range(len(q_scales)):
            axes[2].text(j, i, f'{difference[i, j]:.3f}', 
                       ha='center', va='center', color='black', fontsize=8)
    
    # Добавление цветовых шкал
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('kalman_comparison_grid.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nВизуализация сохранена в файл: kalman_comparison_grid.png")

if __name__ == "__main__":
    print("Запуск сеточного эксперимента...")
    print("Это может занять некоторое время...")
    
    # Запуск эксперимента
    results = run_grid_experiment(n_steps=100, n_state=2, seed=42)
    
    # Создание таблиц
    print("\n" + "="*80)
    print("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА")
    print("="*80)
    
    tables = create_tables(results)
    
    # Сохранение в Excel
    df_akf, df_kf, df_diff = save_tables_to_excel(results, "kalman_grid_comparison.xlsx")
    
    # Визуализация
    visualize_results(results)
    
    # Дополнительный анализ
    print("\n" + "="*80)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("="*80)
    
    # Находим лучшие и худшие случаи
    best_akf_idx = np.unravel_index(np.argmin(results['akf']), results['akf'].shape)
    worst_akf_idx = np.unravel_index(np.argmax(results['akf']), results['akf'].shape)
    
    best_kf_idx = np.unravel_index(np.argmin(results['kf']), results['kf'].shape)
    worst_kf_idx = np.unravel_index(np.argmax(results['kf']), results['kf'].shape)
    
    max_improvement_idx = np.unravel_index(np.argmax(results['kf'] - results['akf']), results['akf'].shape)
    max_degradation_idx = np.unravel_index(np.argmin(results['kf'] - results['akf']), results['akf'].shape)
    
    print(f"Лучший AKF: R_scale={results['r_scales'][best_akf_idx[0]]}, "
          f"Q_scale={results['q_scales'][best_akf_idx[1]]}, "
          f"ARMSE={results['akf'][best_akf_idx]:.6f}")
    
    print(f"Лучший KF: R_scale={results['r_scales'][best_kf_idx[0]]}, "
          f"Q_scale={results['q_scales'][best_kf_idx[1]]}, "
          f"ARMSE={results['kf'][best_kf_idx]:.6f}")
    
    print(f"Максимальное улучшение AKF: R_scale={results['r_scales'][max_improvement_idx[0]]}, "
          f"Q_scale={results['q_scales'][max_improvement_idx[1]]}, "
          f"Улучшение={(results['kf'] - results['akf'])[max_improvement_idx]:.6f}")
    
    # Статистика
    improvement = results['kf'] - results['akf']
    positive_improvements = improvement[improvement > 0]
    negative_improvements = improvement[improvement < 0]
    
    print(f"\nСтатистика:")
    print(f"AKF лучше в {len(positive_improvements)}/{improvement.size} случаях "
          f"({len(positive_improvements)/improvement.size*100:.1f}%)")
    print(f"KF лучше в {len(negative_improvements)}/{improvement.size} случаях "
          f"({len(negative_improvements)/improvement.size*100:.1f}%)")
    print(f"Среднее улучшение при лучшем AKF: {np.mean(positive_improvements):.6f}")
    print(f"Среднее ухудшение при лучшем KF: {np.mean(negative_improvements):.6f}")