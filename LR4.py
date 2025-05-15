import gymnasium as gym
import numpy as np
from pprint import pprint


class PolicyIterationAgent:
    def __init__(self, env):
        self.env = env
        # Пространство состояний (500 для Taxi-v3)
        self.observation_dim = env.observation_space.n
        # Пространство действий (6 для Taxi-v3: north, south, east, west, pickup, dropoff)
        self.actions_variants = np.array([0, 1, 2, 3, 4, 5])
        # Инициализация политики (равномерная вероятность для всех действий)
        self.policy_probs = np.full((self.observation_dim, len(self.actions_variants)), 1.0 / len(self.actions_variants))
        # Начальные значения функции ценности
        self.state_values = np.zeros(shape=(self.observation_dim))
        # Параметры алгоритма
        self.maxNumberOfIterations = 1000
        self.theta = 1e-6
        self.gamma = 0.99

    def print_policy(self):
        '''
        Вывод политики в читаемом формате
        '''
        print('Стратегия:')
        pprint(self.policy_probs)

    def policy_evaluation(self):
        '''
        Оценка текущей политики
        '''
        valueFunctionVector = self.state_values.copy()
        for _ in range(self.maxNumberOfIterations):
            valueFunctionVectorNextIteration = np.zeros(shape=(self.observation_dim))
            for state in range(self.observation_dim):
                outerSum = 0
                action_probabilities = self.policy_probs[state]
                for action, prob in enumerate(action_probabilities):
                    innerSum = 0
                    for probability, next_state, reward, terminated, truncated, _ in self.env.P[state][action]:
                        innerSum += probability * (reward + self.gamma * self.state_values[next_state])
                    outerSum += prob * innerSum
                valueFunctionVectorNextIteration[state] = outerSum
            if np.max(np.abs(valueFunctionVectorNextIteration - valueFunctionVector)) < self.theta:
                valueFunctionVector = valueFunctionVectorNextIteration
                break
            valueFunctionVector = valueFunctionVectorNextIteration
        return valueFunctionVector

    def policy_improvement(self):
        '''
        Улучшение политики
        '''
        qvaluesMatrix = np.zeros((self.observation_dim, len(self.actions_variants)))
        improvedPolicy = np.zeros((self.observation_dim, len(self.actions_variants)))
        for state in range(self.observation_dim):
            for action in range(len(self.actions_variants)):
                for probability, next_state, reward, terminated, truncated, _ in self.env.P[state][action]:
                    qvaluesMatrix[state, action] += probability * (reward + self.gamma * self.state_values[next_state])
            bestActionIndex = np.where(qvaluesMatrix[state, :] == np.max(qvaluesMatrix[state, :]))[0]
            improvedPolicy[state, bestActionIndex] = 1 / len(bestActionIndex)
        return improvedPolicy

    def policy_iteration(self, cnt):
        '''
        Основной цикл алгоритма Policy Iteration
        '''
        for i in range(1, cnt + 1):
            self.state_values = self.policy_evaluation()
            new_policy = self.policy_improvement()
            if np.all(new_policy == self.policy_probs):
                print(f'Политика стабилизировалась на шаге {i}')
                break
            self.policy_probs = new_policy
        print(f'Алгоритм выполнился за {i} шагов.')


def play_agent(agent):
    env2 = gym.make('Taxi-v3', render_mode='human')
    state, _ = env2.reset()
    done = False
    total_reward = 0
    while not done:
        action = np.random.choice(len(agent.actions_variants), p=agent.policy_probs[state])
        next_state, reward, terminated, truncated, _ = env2.step(action)
        total_reward += reward
        env2.render()
        state = next_state
        done = terminated or truncated
    print(f"Итоговая награда: {total_reward}")


def main():
    # Создание среды
    env = gym.make('Taxi-v3')
    env.reset()
    # Обучение агента
    agent = PolicyIterationAgent(env)
    print("Начальная политика:")
    agent.print_policy()
    agent.policy_iteration(1000)
    print("Итоговая политика:")
    agent.print_policy()
    # Тестирование обученного агента
    play_agent(agent)


if __name__ == '__main__':
    main()