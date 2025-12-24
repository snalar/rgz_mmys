import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.linalg import expm
import pandas as pd
from datetime import datetime, timedelta

class FinancialAdaptiveFilter:
    """Адаптивный фильтр для финансовой модели"""
    
    def __init__(self, dt=1.0):
        self.dt = dt  # 1 торговый день
        
        # Параметры модели
        self.lambda_param = 0.1  # скорость возврата к среднему
        self.kappa = 0.1         # скорость возврата волатильности
        self.theta = 0.0004      # долгосрочная волатильность (2% годовых)
        self.eta = 0.3           # волатильность волатильности
        
        # Матрица перехода
        self.F = np.array([[1, self.dt, 0],
                           [0, np.exp(-self.lambda_param * self.dt), 0],
                           [0, 0, 1 - self.kappa * self.dt]])
        
        # Матрица измерений
        self.H = np.array([[1, 0, 0]])
        
        # Процессный шум (зависит от волатильности)
        self.Q_base = np.array([[0.0001, 0, 0],
                                [0, 0.0001, 0],
                                [0, 0, 0.000001]])
        
        # Начальные условия
        self.x_est = np.array([[100.0], [0.0], [0.0004]])  # цена 100, доходность 0, волатильность 2%
        self.P = np.diag([1.0, 0.01, 0.0001])
        self.R_est = 0.0004  # начальная оценка R (2%)
        self.alpha = 0.05    # коэффициент адаптации
        
        # История
        self.history_x = []
        self.history_P = []
        self.history_R = []
        self.history_innov = []
        
    def predict(self):
        """Прогноз"""
        # Обновляем Q в зависимости от текущей волатильности
        sigma2 = self.x_est[2, 0]
        self.Q = self.Q_base.copy()
        self.Q[0, 0] = sigma2 * self.dt
        self.Q[1, 1] = sigma2 * (1 - np.exp(-2 * self.lambda_param * self.dt)) / (2 * self.lambda_param)
        self.Q[2, 2] = self.eta**2 * sigma2 * self.dt
        
        self.x_pred = self.F @ self.x_est
        self.P_pred = self.F @ self.P @ self.F.T + self.Q
        
        return self.x_pred, self.P_pred
    
    def update(self, z):
        """Коррекция с адаптацией R"""
        # Инновация
        innovation = z - self.H @ self.x_pred
        
        # Ковариация инновации
        S = self.H @ self.P_pred @ self.H.T + self.R_est
        
        # Коэффициент усиления Калмана
        K = self.P_pred @ self.H.T / S
        
        # Обновление оценки
        self.x_est = self.x_pred + K * innovation
        
        # Обновление ковариации
        I = np.eye(3)
        self.P = (I - K @ self.H) @ self.P_pred @ (I - K @ self.H).T + K * self.R_est * K.T
        
        # Адаптация R
        C = innovation**2 - self.H @ self.P_pred @ self.H.T
        if C > 0:  # избегаем отрицательных значений
            self.R_est = (1 - self.alpha) * self.R_est + self.alpha * C
        
        # Сохраняем историю
        self.history_x.append(self.x_est.copy())
        self.history_P.append(self.P.copy())
        self.history_R.append(self.R_est)
        self.history_innov.append(innovation[0, 0])
        
        return self.x_est, self.P
    
    def step(self, z):
        """Полный шаг"""
        self.predict()
        return self.update(z)

def generate_financial_data(n_days=120):
    """Генерация синтетических финансовых данных"""
    # Генерация истинной цены с разными режимами волатильности
    np.random.seed(42)
    
    prices = np.zeros(n_days)
    returns = np.zeros(n_days)
    true_volatility = np.zeros(n_days)
    
    # Начальные условия
    prices[0] = 100.0
    
    # Режимы волатильности
    volatility_regimes = [
        (0, 30, 0.005),   # низкая волатильность (0.5%)
        (30, 60, 0.02),   # средняя волатильность (2%)
        (60, 90, 0.05),   # высокая волатильность (5%)
        (90, 120, 0.03)   # смешанный режим (3%)
    ]
    
    # Генерация истинной цены
    for start, end, vol in volatility_regimes:
        for i in range(start, min(end, n_days)):
            true_volatility[i] = vol
            returns[i] = np.random.normal(0, vol)
            if i > 0:
                prices[i] = prices[i-1] * (1 + returns[i])
    
    # Генерация измерений с разной точностью
    measurements = np.zeros(n_days)
    measurement_noise = np.zeros(n_days)
    
    # Изменение точности измерений
    for i in range(n_days):
        if i < 30:
            noise_std = 0.005 * prices[i]  # 0.5%
        elif i < 60:
            noise_std = 0.02 * prices[i]   # 2%
        elif i < 90:
            noise_std = 0.05 * prices[i]   # 5%
        else:
            # Случайные изменения точности
            noise_std = np.random.uniform(0.01, 0.04) * prices[i]
        
        measurement_noise[i] = noise_std
        measurements[i] = prices[i] + np.random.normal(0, noise_std)
    
    return prices, measurements, true_volatility, measurement_noise, returns

def plot_results():
    """Построение графиков результатов"""
    
    # Генерация данных
    n_days = 120
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n_days)]
    
    true_prices, measurements, true_vol, meas_noise, returns = generate_financial_data(n_days)
    
    # Запуск адаптивного фильтра
    filter_adaptive = FinancialAdaptiveFilter()
    adaptive_estimates = np.zeros(n_days)
    adaptive_vol = np.zeros(n_days)
    adaptive_R = np.zeros(n_days)
    adaptive_errors = np.zeros(n_days)
    
    for i in range(n_days):
        z = np.array([[measurements[i]]])
        x_est, _ = filter_adaptive.step(z)
        adaptive_estimates[i] = x_est[0, 0]
        adaptive_vol[i] = np.sqrt(x_est[2, 0])
        adaptive_R[i] = filter_adaptive.R_est
        adaptive_errors[i] = true_prices[i] - x_est[0, 0]
    
    # Запуск НЕадаптивного фильтра для сравнения
    filter_nonadaptive = FinancialAdaptiveFilter()
    filter_nonadaptive.alpha = 0.0  # выключаем адаптацию
    nonadaptive_estimates = np.zeros(n_days)
    nonadaptive_errors = np.zeros(n_days)
    
    for i in range(n_days):
        z = np.array([[measurements[i]]])
        x_est, _ = filter_nonadaptive.step(z)
        nonadaptive_estimates[i] = x_est[0, 0]
        nonadaptive_errors[i] = true_prices[i] - x_est[0, 0]
    
    # Создание графиков
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Цена и оценки
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(dates, true_prices, 'b-', linewidth=2, label='Истинная цена')
    ax1.plot(dates, measurements, 'r.', markersize=2, alpha=0.5, label='Измерения')
    ax1.plot(dates, adaptive_estimates, 'g-', linewidth=1.5, label='Адаптивная оценка')
    ax1.plot(dates, nonadaptive_estimates, 'm--', linewidth=1, alpha=0.7, label='Неадаптивная оценка')
    ax1.set_xlabel('Дата')
    ax1.set_ylabel('Цена')
    ax1.set_title('Цена актива и оценки фильтра')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Добавляем области разных режимов
    colors = ['lightgreen', 'yellow', 'lightcoral', 'lightblue']
    regimes = [(0, 30), (30, 60), (60, 90), (90, 120)]
    for i, (start, end) in enumerate(regimes):
        rect = Rectangle((dates[start], ax1.get_ylim()[0]), 
                        timedelta(days=end-start), 
                        ax1.get_ylim()[1] - ax1.get_ylim()[0],
                        alpha=0.2, facecolor=colors[i])
        ax1.add_patch(rect)
    
    # 2. Ошибки оценки
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(dates, adaptive_errors, 'g-', linewidth=1.5, label='Адаптивный фильтр')
    ax2.plot(dates, nonadaptive_errors, 'm--', linewidth=1, label='Неадаптивный фильтр')
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.fill_between(dates, -np.sqrt(adaptive_R), np.sqrt(adaptive_R), 
                     alpha=0.2, color='green', label='±√R адаптивный')
    ax2.set_xlabel('Дата')
    ax2.set_ylabel('Ошибка оценки')
    ax2.set_title('Ошибки оценки цены')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Волатильность
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(dates, true_vol, 'b-', linewidth=2, label='Истинная волатильность')
    ax3.plot(dates, adaptive_vol, 'g-', linewidth=1.5, label='Оценка волатильности')
    ax3.set_xlabel('Дата')
    ax3.set_ylabel('Волатильность')
    ax3.set_title('Оценка волатильности')
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Адаптация R
    ax4 = plt.subplot(3, 3, 4)
    true_R = (meas_noise / true_prices)**2
    ax4.plot(dates, np.sqrt(true_R), 'b-', linewidth=2, label='Истинное σ измерений')
    ax4.plot(dates, np.sqrt(adaptive_R), 'g-', linewidth=1.5, label='Оценка σ измерений')
    ax4.set_xlabel('Дата')
    ax4.set_ylabel('СКО измерения')
    ax4.set_title('Адаптация ковариации шума измерений R')
    ax4.legend(loc='upper left', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. Распределение ошибок
    ax5 = plt.subplot(3, 3, 5)
    bins = np.linspace(-10, 10, 40)
    ax5.hist(adaptive_errors, bins=bins, alpha=0.7, color='green', 
             label=f'Адаптивный: μ={np.mean(adaptive_errors):.3f}, σ={np.std(adaptive_errors):.3f}')
    ax5.hist(nonadaptive_errors, bins=bins, alpha=0.5, color='magenta',
             label=f'Неадаптивный: μ={np.mean(nonadaptive_errors):.3f}, σ={np.std(nonadaptive_errors):.3f}')
    ax5.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Ошибка')
    ax5.set_ylabel('Частота')
    ax5.set_title('Распределение ошибок оценки')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # 6. Доходность и инновации
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(dates, returns * 100, 'b-', linewidth=1, alpha=0.7, label='Доходность (%)')
    ax6.plot(dates, filter_adaptive.history_innov, 'r-', linewidth=0.5, alpha=0.7, label='Инновации')
    ax6.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax6.set_xlabel('Дата')
    ax6.set_ylabel('Значение')
    ax6.set_title('Доходность и инновации фильтра')
    ax6.legend(loc='upper left', fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    # 7. Накопленная ошибка
    ax7 = plt.subplot(3, 3, 7)
    cumulative_adaptive = np.cumsum(np.abs(adaptive_errors))
    cumulative_nonadaptive = np.cumsum(np.abs(nonadaptive_errors))
    ax7.plot(dates, cumulative_adaptive, 'g-', linewidth=2, label='Адаптивный фильтр')
    ax7.plot(dates, cumulative_nonadaptive, 'm--', linewidth=2, label='Неадаптивный фильтр')
    ax7.set_xlabel('Дата')
    ax7.set_ylabel('Накопленная ошибка')
    ax7.set_title('Накопленная абсолютная ошибка')
    ax7.legend(loc='upper left', fontsize=8)
    ax7.grid(True, alpha=0.3)
    
    # 8. Соотношение сигнал/шум
    ax8 = plt.subplot(3, 3, 8)
    signal_to_noise = adaptive_vol / np.sqrt(adaptive_R)
    ax8.plot(dates, signal_to_noise, 'purple', linewidth=2)
    ax8.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='SNR=1')
    ax8.set_xlabel('Дата')
    ax8.set_ylabel('SNR')
    ax8.set_title('Соотношение сигнал/шум (волатильность/шум измерений)')
    ax8.legend(loc='upper left', fontsize=8)
    ax8.grid(True, alpha=0.3)
    
    # 9. Вероятностные интервалы
    ax9 = plt.subplot(3, 3, 9)
    conf_level = 1.96  # 95% доверительный интервал
    adaptive_std = np.sqrt(np.array(filter_adaptive.history_P)[:, 0, 0])
    
    ax9.fill_between(dates, 
                    adaptive_estimates - conf_level * adaptive_std,
                    adaptive_estimates + conf_level * adaptive_std,
                    alpha=0.3, color='green', label='95% ДИ адаптивный')
    
    ax9.plot(dates, true_prices, 'b-', linewidth=1, label='Истинная цена')
    ax9.plot(dates, adaptive_estimates, 'g-', linewidth=1.5, label='Оценка')
    
    # Подсчет покрытия
    coverage = np.mean(
        (true_prices >= adaptive_estimates - conf_level * adaptive_std) & 
        (true_prices <= adaptive_estimates + conf_level * adaptive_std)
    ) * 100
    
    ax9.set_xlabel('Дата')
    ax9.set_ylabel('Цена')
    ax9.set_title(f'Доверительные интервалы (покрытие: {coverage:.1f}%)')
    ax9.legend(loc='upper left', fontsize=8)
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Вывод статистики
    print("=" * 60)
    print("СТАТИСТИКА РЕЗУЛЬТАТОВ ФИЛЬТРАЦИИ")
    print("=" * 60)
    print(f"Средняя абсолютная ошибка:")
    print(f"  Адаптивный фильтр: {np.mean(np.abs(adaptive_errors)):.4f}")
    print(f"  Неадаптивный фильтр: {np.mean(np.abs(nonadaptive_errors)):.4f}")
    print(f"  Улучшение: {(1 - np.mean(np.abs(adaptive_errors))/np.mean(np.abs(nonadaptive_errors)))*100:.1f}%")
    
    print(f"\nСКО ошибки:")
    print(f"  Адаптивный: {np.std(adaptive_errors):.4f}")
    print(f"  Неадаптивный: {np.std(nonadaptive_errors):.4f}")
    
    print(f"\nМаксимальная ошибка:")
    print(f"  Адаптивный: {np.max(np.abs(adaptive_errors)):.4f}")
    print(f"  Неадаптивный: {np.max(np.abs(nonadaptive_errors)):.4f}")
    
    # Качество адаптации R
    R_error = np.mean(np.abs(np.sqrt(true_R[30:]) - np.sqrt(adaptive_R[30:])))
    print(f"\nСредняя ошибка оценки σ измерений: {R_error:.4f}")
    
    print("\n" + "=" * 60)
    
    plt.show()
    
    # Дополнительный график - сравнение по режимам
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
    regime_names = ['Низкая волатильность', 'Средняя волатильность', 
                   'Высокая волатильность', 'Смешанный режим']
    
    for idx, (start, end) in enumerate(regimes):
        row = idx // 2
        col = idx % 2
        ax = axes2[row, col]
        
        regime_adaptive_errors = adaptive_errors[start:end]
        regime_nonadaptive_errors = nonadaptive_errors[start:end]
        
        x_pos = np.arange(len(regime_adaptive_errors))
        width = 0.35
        
        ax.bar(x_pos - width/2, np.abs(regime_adaptive_errors), width, 
               label='Адаптивный', alpha=0.7, color='green')
        ax.bar(x_pos + width/2, np.abs(regime_nonadaptive_errors), width, 
               label='Неадаптивный', alpha=0.7, color='magenta')
        
        ax.set_xlabel('День в режиме')
        ax.set_ylabel('Абсолютная ошибка')
        ax.set_title(f'{regime_names[idx]}')
        if idx == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Сравнение ошибок по режимам волатильности', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

# Запуск визуализации
if __name__ == "__main__":
    plot_results()