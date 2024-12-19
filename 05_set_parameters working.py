import numpy as np
from simulator import Simulator
from pathlib import Path
import os
import pinocchio as pin
import matplotlib.pyplot as plt

#Создаем модель робота для симулятора и пночио
current_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(current_dir, "robots/universal_robots_ur5e/ur5e.xml")
model = pin.buildModelFromMJCF(xml_path)
data = model.createData()

#Создаем контейнеры типа список для логирования ошибок
q_list = []
dq_list = []
error_list = []
control_list = []

#создаем контейнер для хранения управления на прошлом шаге для последюущей фильтрации
control_old = np.zeros(6)

#режим работы робота: робатсное управление или PD контрол
mode = "robust"
#применять ли фильтрацию управления
filter = False

#функция нахождения скользящей составляющей управления
def findVs(M_hat, S):
    
    sigma_max = 5 #максимальное сингулярное значение матрицы
    k = 1000
    eps = np.array([200, 200, 100, 100, 10, 10],dtype = float) #здесь размеры трубок заданы разными для разных суставов
    
    eps*=100
    
    M_inv = np.linalg.pinv(M_hat) #псевдообратная матрица
    
    s_norm = np.linalg.norm(S) * np.ones(6) #вычисление составляющей S
    
    for i, e in enumerate(eps): #условие попадание в трубку, если оно выполняется то работает скользящее управление
        if(s_norm[i]<=e):
            s_norm[i] = e
        
    return k * M_inv @ (S / s_norm) / sigma_max


def joint_controller(q: np.ndarray, dq: np.ndarray, t: float) -> np.ndarray:
    """Joint space PD controller.
    
    Args:
        q: Current joint positions [rad]
        dq: Current joint velocities [rad/s]
        t: Current simulation time [s]
        
    Returns:
        tau: Joint torques command [Nm]
    """
    # с помощью пиночео вычисляем основные параметры манипулятора для уравнения Эйлера-Лагранжа
    pin.computeAllTerms(model, data, q, dq) 
    pin.forwardKinematics(model, data, q, dq)

    M = data.M #Инерциальная матрица
    
    '''здесь и далее для некоторых составляющих применяется понятия "ОЦЕНКА" что говорит о неточности,
    для симулирования неточности оценки матриц в управлении чуть искажены относительно реальных'''
    M_hat = 0.99*M
    
    g = data.g #Сила тяжести
    g_hat = 1.01*g
    
    C = data.C #Кореолисова матрица
    C_hat = 0.99*C
    
    #составляющие демпфирования и кулоновского трения для каждого сустава,
    #Так же обратим внимание что оценки отличаются от реальных показаний 
    damping_hat = np.array([0.6, 0.4, 0.6, 0.15, 0.9, 0.12])  # Nm/rad/s
    friction_hat = np.array([0.6, 0.4, 0.57, 0.14, 0.78, 0.20])  # Nm
    
    # Уставка для суставов. Здесь скорости и ускорения равны нулю, то есть путь от точки до точки
    q0 = np.array([-0.7, -1.3, 1., 0, 0, 0])
    dq0 = np.zeros(6)
    ddq0 = np.zeros(6)
    
    
    if mode == "robust":
        
        global control_old
        #Задаем вектор лямбд
        L = np.array([600,800,400,200,100,10],dtype=float)
        
        #Поиск ошибки по положению и скорости
        q_err = q0 - q
        dq_err = dq0 - dq
        
        #вычисление составляющей S
        S = dq_err + L * q_err
        
        #Составляющая управления Vs вычисляется в отдельной функции
        Vs = findVs(M_hat,S)
        
        #Управление есть сумма скользящего и наминального управления
        V = ddq0 + L*dq_err + Vs
        
        #Общий закон управления с учетом инерции, силы тяжести, Кореолисовых сил, демпфирования и трения
        U = M_hat @ V + (C_hat + damping_hat) @ dq + g_hat + friction_hat * np.sign(dq)
        
        if filter:
            U = U + (U - control_old)*0.51
            
            control_old = U 
        
        #логирование ошибок, положения и управления
        q_list.append(q)
        dq_list.append(dq)
        error_list.append(q_err)
        control_list.append(U)
        
        print(q)
        
        return U
    elif mode == "PD":
        
        #коэффициенты управления для П и Д регулятора
        Kp = np.array([100,100,100,5,5,5], dtype = float)
        Kd = np.array([20,20,20,100,100,100], dtype=float)
        
        #посик ошибки 
        q_err = q0 - q
        dq_err = dq0 - dq
        
        #вычисление управления 
        V =  dq_err + Kp * q_err + Kd * dq_err
        
        #Общий закон управления с учетом инерции, силы тяжести, Кореолисовых сил, демпфирования и трения
        U = M_hat @ V + (C_hat + damping_hat) @ dq + g_hat + friction_hat * np.sign(dq)
        
        #Логирование положения, ошибок и управления 
        q_list.append(q)
        dq_list.append(dq)
        error_list.append(q_err)
        control_list.append(U)
        
        return U

def main():
    # создание папки для хранения видео
    Path("logs/videos").mkdir(parents=True, exist_ok=True)
    
    # Инцииализация симулятора
    sim = Simulator(
        xml_path="robots/universal_robots_ur5e/scene.xml",
        enable_task_space=False,  
        show_viewer=True,
        record_video=True,
        video_path="logs/videos/08_parameters.mp4",
        fps=30,
        width=1920,
        height=1080
    )
    
    # Отметим что значения демпфинга отличаются от того, что оценивается в управлении
    damping = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])  # Nm/rad/s
    sim.set_joint_damping(damping)
    
    # Отметим что значения трения отличаются от того что оценивается в управлении
    friction = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])  # Nm
    sim.set_joint_friction(friction)
    
    # свойства инструмента
    ee_name = "end_effector"
    
    original_props = sim.get_body_properties(ee_name)
    print("\nOriginal end-effector properties:")
    print(f"Mass: {original_props['mass']:.3f} kg")
    print(f"Inertia:\n{original_props['inertia']}")
    
    # Устанавливаем инертность инструмента
    sim.modify_body_properties(ee_name, mass=2)
    # Печатаем свойства 
    props = sim.get_body_properties(ee_name)
    print("\nModified end-effector properties:")
    print(f"Mass: {props['mass']:.3f} kg")
    print(f"Inertia:\n{props['inertia']}")
    
    # Для контроллера выбираем контроллер суставов который ранее написали и запускаем симулятор
    sim.set_controller(joint_controller)
    sim.run(time_limit=5.0)

if __name__ == "__main__":
    main() 

    #для вывода графиков преобразуем контейнер список в массив    
    q_list = np.array(q_list)
    dq_list = np.array(dq_list)
    control = np.array(control_list)
    
    
    #отрисовка и сохранение графиков
    plt.figure(figsize=(10, 6))
    for i in range(q_list.shape[1]):
        plt.plot(q_list[:, i], label=f'Joint {i+1}')
    plt.xlabel('Time')
    plt.ylabel('Joint Positions [rad]')
    plt.title('Joint Positions')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/final_positions.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    for i in range(dq_list.shape[1]):
        plt.plot(dq_list[:, i], label=f'Joint {i+1}')
    plt.xlabel('Time')
    plt.ylabel('Joint Velocities [rad/s]')
    plt.title('Joint Velocities')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/final_velocities.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    for i in range(control.shape[1]):
        plt.plot(control[:, i], label=f'Joint {i+1}')
    plt.xlabel('Time')
    plt.ylabel('Control')
    plt.title('Control')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/final_controls.png')
    plt.close()    
    