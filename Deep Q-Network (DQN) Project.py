import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# 創建環境
env = gym.make('CartPole-v1')

# 建立DQN模型
def build_model(state_shape, action_shape):
    model = tf.keras.Sequential()
    model.add(layers.Dense(24, input_shape=state_shape, activation='relu'))
    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dense(action_shape, activation='linear'))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='mse')
    return model

# 設置參數
state_shape = env.observation_space.shape
action_shape = env.action_space.n
model = build_model(state_shape, action_shape)
target_model = build_model(state_shape, action_shape)
target_model.set_weights(model.get_weights())

gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 64
memory = []

# 訓練DQN
def train_dqn(episodes):
    for episode in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_shape[0]])
        done = False
        while not done:
            if np.random.rand() <= epsilon:
                action = np.random.choice(action_shape)
            else:
                action = np.argmax(model.predict(state))
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_shape[0]])
            memory.append((state, action, reward, next_state, done))
            state = next_state
            if done:
                print(f"Episode: {episode+1}/{episodes}, Score: {reward}")
                if epsilon > epsilon_min:
                    epsilon *= epsilon_decay
            if len(memory) > batch_size:
                minibatch = np.random.choice(memory, batch_size)
                for state, action, reward, next_state, done in minibatch:
                    target = reward
                    if not done:
                        target += gamma * np.amax(target_model.predict(next_state))
                    target_f = model.predict(state)
                    target_f[0][action] = target
                    model.fit(state, target_f, epochs=1, verbose=0)
                target_model.set_weights(model.get_weights())

train_dqn(1000)