import numpy as np
import matplotlib.pyplot as plt

# Criando um array de tempo de -2 a 5 com passo de 0.01
t = np.arange(-2, 5, 0.01)

# Definindo a função degrau unitário
def u(t):
    return 1.0 * (t >= 0)

# Calculando a função e^u(t)
e_ut = np.exp(u(t))

# Plotando o gráfico
plt.plot(t, e_ut, label='e^u(t)')
plt.xlabel('Tempo')
plt.ylabel('e^u(t)')
plt.title('Gráfico de e^u(t)')
plt.grid(True)
plt.legend()
plt.show()
