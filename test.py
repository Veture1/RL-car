import time
import pickle
import gym
import numpy as np

def transform_state(state):
    pos, v = state
    pos_low, v_low = env.observation_space.low
    pos_high, v_high = env.observation_space.high

    a = 50 * (pos-pos_low) / (pos_high-pos_low)
    b = 50 * (v-v_low) / (v_high-v_low)

    return int(a), int(b)

if __name__ == "__main__":
    model_name = input("Input the model name you want to test: ")
    # load the model
    with open(model_name, 'rb') as f:
        Q = pickle.load(f)
        print("Model loaded!")

    env = gym.make("MountainCar-v0", render_mode="human")
    state, _ = env.reset()

    score = 0

    while True:
        env.render()
        time.sleep(0.01)
        s = transform_state(state)
        a = np.argmax(Q[s]) if s in Q else 0
        s, reward, done, truncated, _ = env.step(a)
        score += reward

        if done or truncated:
            print(f"Score: {score}")
            break


    env.close()