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
