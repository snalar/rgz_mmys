import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.linalg import expm
import pandas as pd
from datetime import datetime, timedelta

class ImprovedFinancialAdaptiveFilter:
    """УЛУЧШЕННЫЙ адаптивный фильтр для финансовой модели"""
    
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
        
        # РАЗНЫЕ коэффициенты адаптации для разных параметров
        self.alpha_R = 0.15    # быстрая адаптация R (шум измерений)
        self.alpha_Q = 0.05    # медленная адаптация Q (волатильность процесса)
        self.alpha_vol = 0.02  # очень медленная адаптация волатильности
        
        # Инициализация R на основе первых измерений
        self.R_est = 0.0004  # начальная оценка R (2%)
        self.R_min = 0.000001  # минимальное R (0.01%)
        self.R_max = 0.01     # максимальное R (10%)
        
        # Для адаптивного alpha
        self.innovation_history = []
        self.max_history = 20
        
        # История
        self.history_x = []
        self.history_P = []
        self.history_R = []
        self.history_innov = []
        self.history_alpha = []
        self.history_vol = []
        
    def calculate_adaptive_alpha(self, innovation):
        """Рассчитывает адаптивный коэффициент забывания на основе стабильности инноваций"""
        self.innovation_history.append(abs(float(innovation)))
        if len(self.innovation_history) > self.max_history:
            self.innovation_history.pop(0)
        
        if len(self.innovation_history) < 5:
            return self.alpha_R, self.alpha_Q, self.alpha_vol
        
        # Анализ стабильности инноваций
        innov_array = np.array(self.innovation_history)
        innov_mean = np.mean(innov_array)
        innov_std = np.std(innov_array)
        
        if innov_mean == 0:
            return self.alpha_R, self.alpha_Q, self.alpha_vol
        
        # Коэффициент вариации инноваций
        cv = innov_std / innov_mean
        
        # Динамическая настройка alpha:
        # - Если инновации стабильны (низкий cv) -> медленная адаптация
        # - Если инновации нестабильны (высокий cv) -> быстрая адаптация
        adaptive_factor = 1 + cv
        
        alpha_R_adaptive = min(0.3, max(0.05, self.alpha_R * adaptive_factor))
        alpha_Q_adaptive = min(0.1, max(0.01, self.alpha_Q * adaptive_factor))
        alpha_vol_adaptive = min(0.05, max(0.005, self.alpha_vol * adaptive_factor))
        
        return alpha_R_adaptive, alpha_Q_adaptive, alpha_vol_adaptive
    
    def predict(self):
        """Прогноз"""
        # Обновляем Q в зависимости от текущей волатильности
        sigma2 = max(0.000001, self.x_est[2, 0])  # защита от отрицательных значений
        self.Q = self.Q_base.copy()
        self.Q[0, 0] = sigma2 * self.dt
        self.Q[1, 1] = sigma2 * (1 - np.exp(-2 * self.lambda_param * self.dt)) / (2 * self.lambda_param)
        self.Q[2, 2] = self.eta**2 * sigma2 * self.dt
        
        self.x_pred = self.F @ self.x_est
        self.P_pred = self.F @ self.P @ self.F.T + self.Q
        
        return self.x_pred, self.P_pred
    
    def update(self, z):
        """Коррекция с УЛУЧШЕННОЙ адаптацией"""
        # Инновация
        innovation = z - self.H @ self.x_pred
        
        # Ковариация инновации
        S = self.H @ self.P_pred @ self.H.T + self.R_est
        
        # Коэффициент усиления Калмана
        K = self.P_pred @ self.H.T / S
        
        # Обновление оценки
        self.x_est = self.x_pred + K * innovation
        
        # Обновление ковариации (формула Джозефа для устойчивости)
        I = np.eye(3)
        self.P = (I - K @ self.H) @ self.P_pred @ (I - K @ self.H).T + K * self.R_est * K.T
        
        # УЛУЧШЕННАЯ адаптация параметров
        
        # 1. Адаптация R (шум измерений)
        innovation_var = float(innovation**2)
        predicted_var = float(self.H @ self.P_pred @ self.H.T)
        C_R = innovation_var - predicted_var
        
        # Динамические alpha
        alpha_R, alpha_Q, alpha_vol = self.calculate_adaptive_alpha(innovation)
        self.history_alpha.append(alpha_R)
        
        if abs(C_R) > 1e-8:
            # Адаптируем R в обе стороны
            R_new = (1 - alpha_R) * self.R_est + alpha_R * abs(C_R)
            # Ограничения
            self.R_est = max(self.R_min, min(R_new, self.R_max))
        
        # 2. Адаптация волатильности (только если уверены)
        if len(self.history_innov) > 10:
            recent_innovations = np.array(self.history_innov[-10:])
            recent_innov_var = np.var(recent_innovations)
            
            # Плавная адаптация волатильности
            if recent_innov_var > 0:
                vol_estimate = np.sqrt(recent_innov_var) / 100  # нормализация
                current_vol = self.x_est[2, 0]
                new_vol = (1 - alpha_vol) * current_vol + alpha_vol * vol_estimate
                # Ограничения на волатильность
                self.x_est[2, 0] = max(0.000001, min(new_vol, 0.01))
        
        # Сохраняем историю
        self.history_x.append(self.x_est.copy())
        self.history_P.append(self.P.copy())
        self.history_R.append(self.R_est)
        self.history_innov.append(innovation[0, 0])
        self.history_vol.append(self.x_est[2, 0])
        
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

def plot_improved_results():
    """Построение графиков результатов УЛУЧШЕННОГО алгоритма"""
    
    # Генерация данных
    n_days = 120
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n_days)]
    
    true_prices, measurements, true_vol, meas_noise, returns = generate_financial_data(n_days)
    
    # Запуск УЛУЧШЕННОГО адаптивного фильтра
    filter_adaptive = ImprovedFinancialAdaptiveFilter()
    adaptive_estimates = np.zeros(n_days)
    adaptive_vol = np.zeros(n_days)
    adaptive_R = np.zeros(n_days)
    adaptive_errors = np.zeros(n_days)
    adaptive_alpha = np.zeros(n_days)
    
    for i in range(n_days):
        z = np.array([[measurements[i]]])
        x_est, _ = filter_adaptive.step(z)
        adaptive_estimates[i] = x_est[0, 0]
        adaptive_vol[i] = np.sqrt(x_est[2, 0])
        adaptive_R[i] = filter_adaptive.R_est
        adaptive_errors[i] = true_prices[i] - x_est[0, 0]
        if i > 0 and len(filter_adaptive.history_alpha) > 0:
            adaptive_alpha[i] = filter_adaptive.history_alpha[-1]
    
    # Запуск НЕадаптивного фильтра для сравнения
    filter_nonadaptive = ImprovedFinancialAdaptiveFilter()
    filter_nonadaptive.alpha_R = 0.0  # выключаем адаптацию
    filter_nonadaptive.alpha_Q = 0.0
    filter_nonadaptive.alpha_vol = 0.0
    nonadaptive_estimates = np.zeros(n_days)
    nonadaptive_errors = np.zeros(n_days)
    
    for i in range(n_days):
        z = np.array([[measurements[i]]])
        x_est, _ = filter_nonadaptive.step(z)
        nonadaptive_estimates[i] = x_est[0, 0]
        nonadaptive_errors[i] = true_prices[i] - x_est[0, 0]
    
    # Создание графиков
    fig = plt.figure(figsize=(18, 14))
    
    # 1. Цена и оценки
    ax1 = plt.subplot(4, 3, 1)
    ax1.plot(dates, true_prices, 'b-', linewidth=2, label='Истинная цена')
    ax1.plot(dates, measurements, 'r.', markersize=2, alpha=0.3, label='Измерения')
    ax1.plot(dates, adaptive_estimates, 'g-', linewidth=1.5, label='Улучшенный адаптивный')
    ax1.plot(dates, nonadaptive_estimates, 'm--', linewidth=1, alpha=0.7, label='Неадаптивный')
    ax1.set_xlabel('Дата')
    ax1.set_ylabel('Цена')
    ax1.set_title('Цена актива и оценки фильтра (улучшенный)')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Добавляем области разных режимов
    colors = ['lightgreen', 'yellow', 'lightcoral', 'lightblue']
    regimes = [(0, 30), (30, 60), (60, 90), (90, 120)]
    for i, (start, end) in enumerate(regimes):
        rect = Rectangle((dates[start], ax1.get_ylim()[0]), 
                        timedelta(days=end-start), 
                        ax1.get_ylim()[1] - ax1.get_ylim()[0],
                        alpha=0.15, facecolor=colors[i], edgecolor='gray', linewidth=0.5)
        ax1.add_patch(rect)
    
    # 2. Ошибки оценки
    ax2 = plt.subplot(4, 3, 2)
    ax2.plot(dates, adaptive_errors, 'g-', linewidth=1.5, label='Улучшенный адаптивный')
    ax2.plot(dates, nonadaptive_errors, 'm--', linewidth=1, label='Неадаптивный')
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Полоса ±√R адаптивного
    adaptive_R_std = np.sqrt(adaptive_R)
    ax2.fill_between(dates, -adaptive_R_std * 100, adaptive_R_std * 100, 
                     alpha=0.15, color='green', label='±√R адаптивный')
    
    ax2.set_xlabel('Дата')
    ax2.set_ylabel('Ошибка оценки')
    ax2.set_title('Ошибки оценки цены')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Волатильность
    ax3 = plt.subplot(4, 3, 3)
    ax3.plot(dates, true_vol, 'b-', linewidth=2, label='Истинная волатильность')
    ax3.plot(dates, adaptive_vol, 'g-', linewidth=1.5, label='Оценка волатильности')
    ax3.plot(dates, np.sqrt(filter_adaptive.history_vol), 'c--', linewidth=1, alpha=0.7, label='Внутренняя оценка')
    ax3.set_xlabel('Дата')
    ax3.set_ylabel('Волатильность')
    ax3.set_title('Оценка волатильности (улучшенная)')
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Адаптация R
    ax4 = plt.subplot(4, 3, 4)
    true_R = (meas_noise / true_prices)**2
    ax4.plot(dates, np.sqrt(true_R), 'b-', linewidth=2, label='Истинное σ измерений')
    ax4.plot(dates, np.sqrt(adaptive_R), 'g-', linewidth=1.5, label='Оценка σ измерений')
    ax4.set_xlabel('Дата')
    ax4.set_ylabel('СКО измерения')
    ax4.set_title('Адаптация ковариации шума измерений R (улучшенная)')
    ax4.legend(loc='upper left', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. Адаптивный коэффициент забывания
    ax5 = plt.subplot(4, 3, 5)
    ax5.plot(dates, adaptive_alpha, 'purple', linewidth=2)
    ax5.axhline(y=filter_adaptive.alpha_R, color='k', linestyle='--', alpha=0.5, 
               label=f'Базовый α={filter_adaptive.alpha_R}')
    ax5.set_xlabel('Дата')
    ax5.set_ylabel('α')
    ax5.set_title('Адаптивный коэффициент забывания')
    ax5.legend(loc='upper left', fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # 6. Распределение ошибок
    ax6 = plt.subplot(4, 3, 6)
    bins = np.linspace(-10, 10, 40)
    
    adaptive_mean = np.mean(adaptive_errors)
    adaptive_std = np.std(adaptive_errors)
    nonadaptive_mean = np.mean(nonadaptive_errors)
    nonadaptive_std = np.std(nonadaptive_errors)
    
    ax6.hist(adaptive_errors, bins=bins, alpha=0.7, color='green', density=True,
             label=f'Улучшенный: μ={adaptive_mean:.3f}, σ={adaptive_std:.3f}')
    ax6.hist(nonadaptive_errors, bins=bins, alpha=0.5, color='magenta', density=True,
             label=f'Неадаптивный: μ={nonadaptive_mean:.3f}, σ={nonadaptive_std:.3f}')
    
    # Нормальные распределения для сравнения
    x = np.linspace(-10, 10, 200)
    ax6.plot(x, 1/(adaptive_std*np.sqrt(2*np.pi))*np.exp(-0.5*((x-adaptive_mean)/adaptive_std)**2),
            'g--', alpha=0.7, linewidth=1.5)
    ax6.plot(x, 1/(nonadaptive_std*np.sqrt(2*np.pi))*np.exp(-0.5*((x-nonadaptive_mean)/nonadaptive_std)**2),
            'm--', alpha=0.7, linewidth=1.5)
    
    ax6.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Ошибка')
    ax6.set_ylabel('Плотность вероятности')
    ax6.set_title('Распределение ошибок оценки (улучшенный)')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    # 7. Доходность и инновации
    ax7 = plt.subplot(4, 3, 7)
    ax7.plot(dates, returns * 100, 'b-', linewidth=1, alpha=0.7, label='Доходность (%)')
    ax7.plot(dates, filter_adaptive.history_innov, 'r-', linewidth=0.5, alpha=0.7, label='Инновации')
    
    # Скользящее среднее инноваций
    if len(filter_adaptive.history_innov) > 5:
        innov_ma = np.convolve(filter_adaptive.history_innov, np.ones(5)/5, mode='valid')
        ax7.plot(dates[2:-2], innov_ma, 'orange', linewidth=1.5, alpha=0.8, label='Среднее инноваций (MA5)')
    
    ax7.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax7.set_xlabel('Дата')
    ax7.set_ylabel('Значение')
    ax7.set_title('Доходность и инновации фильтра')
    ax7.legend(loc='upper left', fontsize=8)
    ax7.grid(True, alpha=0.3)
    
    # 8. Накопленная ошибка
    ax8 = plt.subplot(4, 3, 8)
    cumulative_adaptive = np.cumsum(np.abs(adaptive_errors))
    cumulative_nonadaptive = np.cumsum(np.abs(nonadaptive_errors))
    ax8.plot(dates, cumulative_adaptive, 'g-', linewidth=2, label='Улучшенный адаптивный')
    ax8.plot(dates, cumulative_nonadaptive, 'm--', linewidth=2, label='Неадаптивный')
    
    # Разница в накопленной ошибке
    error_diff = cumulative_nonadaptive - cumulative_adaptive
    ax8.fill_between(dates, cumulative_adaptive, cumulative_nonadaptive, 
                     alpha=0.2, color='green', label='Выигрыш адаптивного')
    
    ax8.set_xlabel('Дата')
    ax8.set_ylabel('Накопленная ошибка')
    ax8.set_title('Накопленная абсолютная ошибка')
    ax8.legend(loc='upper left', fontsize=8)
    ax8.grid(True, alpha=0.3)
    
    # 9. Соотношение сигнал/шум
    ax9 = plt.subplot(4, 3, 9)
    signal_to_noise = adaptive_vol / np.sqrt(adaptive_R)
    ax9.plot(dates, signal_to_noise, 'purple', linewidth=2)
    ax9.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='SNR=1')
    ax9.axhline(y=np.median(signal_to_noise), color='r', linestyle='--', alpha=0.7, 
               label=f'Медиана={np.median(signal_to_noise):.2f}')
    ax9.set_xlabel('Дата')
    ax9.set_ylabel('SNR')
    ax9.set_title('Соотношение сигнал/шум (волатильность/шум измерений)')
    ax9.legend(loc='upper left', fontsize=8)
    ax9.grid(True, alpha=0.3)
    
    # 10. Вероятностные интервалы
    ax10 = plt.subplot(4, 3, 10)
    conf_level = 1.96  # 95% доверительный интервал
    if len(filter_adaptive.history_P) > 0:
        adaptive_std = np.sqrt(np.array(filter_adaptive.history_P)[:, 0, 0])
        
        ax10.fill_between(dates, 
                         adaptive_estimates - conf_level * adaptive_std,
                         adaptive_estimates + conf_level * adaptive_std,
                         alpha=0.3, color='green', label='95% ДИ улучшенный')
    
    ax10.plot(dates, true_prices, 'b-', linewidth=1, alpha=0.7, label='Истинная цена')
    ax10.plot(dates, adaptive_estimates, 'g-', linewidth=1.5, label='Оценка')
    
    # Подсчет покрытия
    if len(filter_adaptive.history_P) > 0:
        coverage = np.mean(
            (true_prices >= adaptive_estimates - conf_level * adaptive_std) & 
            (true_prices <= adaptive_estimates + conf_level * adaptive_std)
        ) * 100
    
    ax10.set_xlabel('Дата')
    ax10.set_ylabel('Цена')
    ax10.set_title(f'Доверительные интервалы (покрытие: {coverage:.1f}%)')
    ax10.legend(loc='upper left', fontsize=8)
    ax10.grid(True, alpha=0.3)
    
    # 11. Относительное улучшение по дням
    ax11 = plt.subplot(4, 3, 11)
    daily_improvement = 100 * (np.abs(nonadaptive_errors) - np.abs(adaptive_errors)) / np.abs(nonadaptive_errors + 1e-10)
    
    # Скользящее среднее улучшения
    ma_window = 7
    if len(daily_improvement) > ma_window:
        improvement_ma = np.convolve(daily_improvement, np.ones(ma_window)/ma_window, mode='valid')
        ax11.plot(dates[ma_window//2:-ma_window//2+1], improvement_ma, 'g-', linewidth=2, 
                 label=f'Скользящее среднее (MA{ma_window})')
    
    ax11.bar(dates, daily_improvement, alpha=0.3, color='green', width=0.8, label='Дневное улучшение')
    ax11.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax11.axhline(y=np.mean(daily_improvement), color='r', linestyle='--', 
                label=f'Среднее: {np.mean(daily_improvement):.1f}%')
    
    ax11.set_xlabel('Дата')
    ax11.set_ylabel('Улучшение (%)')
    ax11.set_title('Ежедневное улучшение адаптивного алгоритма')
    ax11.legend(loc='upper left', fontsize=8)
    ax11.grid(True, alpha=0.3)
    
    # 12. Сравнение ошибок по режимам
    ax12 = plt.subplot(4, 3, 12)
    regime_errors_adaptive = []
    regime_errors_nonadaptive = []
    regime_names = ['Низкая', 'Средняя', 'Высокая', 'Смешанная']
    
    for start, end in regimes:
        regime_errors_adaptive.append(np.mean(np.abs(adaptive_errors[start:end])))
        regime_errors_nonadaptive.append(np.mean(np.abs(nonadaptive_errors[start:end])))
    
    x = np.arange(len(regime_names))
    width = 0.35
    
    ax12.bar(x - width/2, regime_errors_adaptive, width, color='green', alpha=0.7, label='Улучшенный адаптивный')
    ax12.bar(x + width/2, regime_errors_nonadaptive, width, color='magenta', alpha=0.7, label='Неадаптивный')
    
    # Добавление значений улучшения
    for i in range(len(regime_names)):
        improvement = 100 * (regime_errors_nonadaptive[i] - regime_errors_adaptive[i]) / regime_errors_nonadaptive[i]
        ax12.text(i, max(regime_errors_adaptive[i], regime_errors_nonadaptive[i]) + 0.05,
                 f'{improvement:.1f}%', ha='center', fontsize=9, fontweight='bold')
    
    ax12.set_xlabel('Режим волатильности')
    ax12.set_ylabel('Средняя абсолютная ошибка')
    ax12.set_title('Сравнение ошибок по режимам')
    ax12.set_xticks(x)
    ax12.set_xticklabels(regime_names)
    ax12.legend()
    ax12.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Вывод статистики
    print("=" * 70)
    print("СТАТИСТИКА РЕЗУЛЬТАТОВ УЛУЧШЕННОЙ ФИЛЬТРАЦИИ")
    print("=" * 70)
    
    mae_adaptive = np.mean(np.abs(adaptive_errors))
    mae_nonadaptive = np.mean(np.abs(nonadaptive_errors))
    improvement_pct = 100 * (mae_nonadaptive - mae_adaptive) / mae_nonadaptive
    
    print(f"Средняя абсолютная ошибка (MAE):")
    print(f"  Улучшенный адаптивный фильтр: {mae_adaptive:.4f}")
    print(f"  Неадаптивный фильтр: {mae_nonadaptive:.4f}")
    print(f"  Улучшение: {improvement_pct:.1f}%")
    
    print(f"\nСКО ошибки (RMSE):")
    rmse_adaptive = np.sqrt(np.mean(adaptive_errors**2))
    rmse_nonadaptive = np.sqrt(np.mean(nonadaptive_errors**2))
    print(f"  Улучшенный адаптивный: {rmse_adaptive:.4f}")
    print(f"  Неадаптивный: {rmse_nonadaptive:.4f}")
    
    print(f"\nМедианная абсолютная ошибка:")
    print(f"  Улучшенный адаптивный: {np.median(np.abs(adaptive_errors)):.4f}")
    print(f"  Неадаптивный: {np.median(np.abs(nonadaptive_errors)):.4f}")
    
    print(f"\nМаксимальная абсолютная ошибка:")
    print(f"  Улучшенный адаптивный: {np.max(np.abs(adaptive_errors)):.4f}")
    print(f"  Неадаптивный: {np.max(np.abs(nonadaptive_errors)):.4f}")
    
    # Качество адаптации R
    if len(true_R) == len(adaptive_R):
        R_mae = np.mean(np.abs(np.sqrt(true_R[30:]) - np.sqrt(adaptive_R[30:])))
        R_corr = np.corrcoef(np.sqrt(true_R[30:]), np.sqrt(adaptive_R[30:]))[0, 1]
        print(f"\nКачество адаптации R (шум измерений):")
        print(f"  Средняя ошибка σ: {R_mae:.4f}")
        print(f"  Корреляция с истинным σ: {R_corr:.3f}")
    
    # Анализ по режимам
    print(f"\nАнализ по режимам волатильности:")
    for i, (start, end) in enumerate(regimes):
        regime_adaptive = np.mean(np.abs(adaptive_errors[start:end]))
        regime_nonadaptive = np.mean(np.abs(nonadaptive_errors[start:end]))
        regime_improvement = 100 * (regime_nonadaptive - regime_adaptive) / regime_nonadaptive
        print(f"  {regime_names[i]}: {regime_improvement:+.1f}% улучшения")
    
    print("\n" + "=" * 70)
    
    plt.show()
    
    # Дополнительный график - обучение и адаптация
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. Сходимость R
    axes2[0, 0].plot(np.sqrt(adaptive_R), 'g-', linewidth=2, label='Оценка σ')
    axes2[0, 0].plot(np.sqrt(true_R), 'b--', alpha=0.7, label='Истинный σ')
    axes2[0, 0].set_xlabel('Шаг')
    axes2[0, 0].set_ylabel('СКО измерения')
    axes2[0, 0].set_title('Сходимость оценки шума измерений')
    axes2[0, 0].legend()
    axes2[0, 0].grid(True, alpha=0.3)
    
    # 2. Адаптация коэффициента забывания
    if len(filter_adaptive.history_alpha) > 0:
        axes2[0, 1].plot(filter_adaptive.history_alpha, 'purple', linewidth=2)
        axes2[0, 1].axhline(y=filter_adaptive.alpha_R, color='k', linestyle='--', 
                           label=f'Базовый α={filter_adaptive.alpha_R}')
        axes2[0, 1].set_xlabel('Шаг')
        axes2[0, 1].set_ylabel('α')
        axes2[0, 1].set_title('Динамика коэффициента забывания')
        axes2[0, 1].legend()
        axes2[0, 1].grid(True, alpha=0.3)
    
    # 3. Ошибка vs волатильность
    axes2[1, 0].scatter(true_vol[10:], np.abs(adaptive_errors[10:]), alpha=0.5, 
                       c='green', s=20, label='Адаптивный')
    axes2[1, 0].scatter(true_vol[10:], np.abs(nonadaptive_errors[10:]), alpha=0.3, 
                       c='magenta', s=20, label='Неадаптивный')
    axes2[1, 0].set_xlabel('Истинная волатильность')
    axes2[1, 0].set_ylabel('Абсолютная ошибка')
    axes2[1, 0].set_title('Зависимость ошибки от волатильности')
    axes2[1, 0].legend()
    axes2[1, 0].grid(True, alpha=0.3)
    
    # 4. Кумулятивная разница
    axes2[1, 1].plot(np.cumsum(np.abs(nonadaptive_errors) - np.abs(adaptive_errors)), 
                    'g-', linewidth=2)
    axes2[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes2[1, 1].set_xlabel('Шаг')
    axes2[1, 1].set_ylabel('Накопленная разница ошибок')
    axes2[1, 1].set_title('Накопленное преимущество адаптивного фильтра')
    axes2[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Динамика обучения и адаптации улучшенного алгоритма', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

# Запуск улучшенной визуализации
if __name__ == "__main__":
    plot_improved_results()