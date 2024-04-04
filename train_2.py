import os
import sys
import glob
import time
import math
import random
import configparser
from datetime import datetime
from threading import Thread
from collections import deque
from tqdm import tqdm

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.callbacks import TensorBoard

LOG_PATH = "/home/node/Node/Logs"
CARLA_PATH = "/home/node/Node/Tools/Carla"
CONFIG_FILE = "/home/node/Node/Tools/Config.ini"
MAINFRAME_IP = "127.0.0.1"

EPISODES = 100
EPISODE_SECONDS = 10
TARGET_UPDATE_INTERVAL = 5
STATS_UPDATE_INTERVAL = 10

REPLAY_MEMORY_SIZE = 5000
REPLAY_MEMORY_MINIMUM = 1000
MINI_BATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINI_BATCH_SIZE // 4

LEARNING_RATE = 0.001
MEMORY_FRACTION = 0.8
REWARD_MINIMUM = -200
DISCOUNT = 0.99
EPSILON = 1
EPSILON_DECAY = 0.95
EPSILON_MINIMUM = 0.001

SHOW_PREVIEW = False
TIMESTAMP = None
IMAGE_DIMENSIONS = (640, 480)
STEER_AMOUNT = 1.0
FPS = 30


def import_carla():
    print("Importing Carla...")
    try:
        sys.path.append(glob.glob(CARLA_PATH + "/dist/carla-*%d.%d-%s.egg" % (
            3,
            7,
            "win-amd64" if os.name == "nt" else "linux-x86_64"))[0])
    except IndexError:
        print("  Error!")
    print("Done!")


def load_config():
    global MAINFRAME_IP

    print("Loading Config...")
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)

    MAINFRAME_IP = config["Network"]["Mainframe"]

    print("  Mainframe-IP:", MAINFRAME_IP)
    print("Done!")


def main():
    global EPSILON, TIMESTAMP

    episode_rewards = [-200]
    TIMESTAMP = str(datetime.now().strftime("%H-%M-%S-%d-%m-%Y"))

    random.seed(1)
    np.random.seed(1)
    tf.random.set_seed(1)

    agent = Agent()
    env = Env()

    # Thread for training
    train_thread = Thread(target=agent.train_loop, daemon=True)
    train_thread.start()
    while not agent.initialized:
        time.sleep(0.01)
    agent.get_qs(np.ones((IMAGE_DIMENSIONS[1], IMAGE_DIMENSIONS[0], 3)))

    # Train loop
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit="episodes"):
        env.collision_history = []
        agent.tensorboard.step = episode
        current_state = env.reset()

        step = 1
        episode_reward = 0

        while True:
            if np.random.random() > EPSILON:
                action = np.argmax(agent.get_qs(current_state))
            else:
                action = np.random.randint(0, 3)
                time.sleep(1 / FPS)

            # Calculate q-values
            new_state, reward, done = env.step(action)
            episode_reward += reward
            agent.update_memory((current_state, action, reward, new_state, done))
            current_state = new_state
            step += 1

            if done:
                break

        for actor in env.actor_list:
            actor.destroy()

        # Collect metrics
        episode_rewards.append(episode_reward)
        if not episode % STATS_UPDATE_INTERVAL or episode == 1:
            average_reward = sum(episode_rewards[-STATS_UPDATE_INTERVAL:]) / len(
                episode_rewards[-STATS_UPDATE_INTERVAL:])
            min_reward = min(episode_rewards[-STATS_UPDATE_INTERVAL:])
            max_reward = max(episode_rewards[-STATS_UPDATE_INTERVAL:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                           epsilon=EPSILON)
            if min_reward >= REWARD_MINIMUM:
                agent.model.save(LOG_PATH + "/" + TIMESTAMP + "/model")

            if EPSILON > EPSILON_MINIMUM:
                EPSILON *= EPSILON_DECAY
                EPSILON = max(EPSILON_MINIMUM, EPSILON)

        agent.terminate = True
        train_thread.join()
        agent.model.save(LOG_PATH + "/" + TIMESTAMP + "/model")


class Env:
    global MAINFRAME_IP

    def __init__(self):
        self.actor_list = []
        self.collision_history = []

        print("Connecting to Simulation...")
        self.client = carla.Client(MAINFRAME_IP, 2000)
        self.client.set_timeout(5.0)
        print("Done!")

        self.world = self.client.get_world()
        blueprint_library = self.world.get_blueprint_library()

        self.vehicle_bp = blueprint_library.filter("cybertruck")[0]
        self.vehicle = None

        self.camera_bp = blueprint_library.find("sensor.camera.rgb")
        self.camera_bp.set_attribute("image_size_x", str(IMAGE_DIMENSIONS[0]))
        self.camera_bp.set_attribute("image_size_y", str(IMAGE_DIMENSIONS[1]))
        self.camera_bp.set_attribute("fov", "110")
        self.camera = None

        self.sensor_bp = blueprint_library.find("sensor.other.collision")
        self.sensor = None

        self.camera_stream = None
        self.episode_start = None

    def reset(self):
        self.actor_list.clear()
        self.collision_history.clear()

        vehicle_sp = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.vehicle_bp, vehicle_sp)
        self.actor_list.append(self.vehicle)
        print("  Spawned Vehicle:", self.vehicle.type_id)

        camera_sp = carla.Transform(carla.Location(x=3.0, z=0.7))
        self.camera = self.world.spawn_actor(self.camera_bp, camera_sp, attach_to=self.vehicle)
        self.actor_list.append(self.camera)
        print("  Spawned Camera:", self.camera.type_id)

        sensor_sp = camera_sp
        self.sensor = self.world.spawn_actor(self.sensor_bp, sensor_sp, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        print("  Spawned Sensor:", self.sensor.type_id)

        time.sleep(3)

        self.vehicle.apply_control(carla.VehicleControl(throttle=0, brake=0, steer=0))
        self.camera.listen(lambda data: self.process_image(data))
        self.sensor.listen(lambda data: self.process_collision(data))

        while self.camera_stream is None:
            time.sleep(1)

        self.episode_start = time.time()
        return self.camera_stream

    def destroy(self):
        print("Destroying actors...")
        self.camera.destroy()
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
        print("Done!")

    def step(self, action):
        if action == 0:  # Accelerate
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, brake=0, steer=0))
        elif action == 1:  # Steer left
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, brake=0, steer=-1 * STEER_AMOUNT))
        elif action == 2:  # Steer right
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, brake=0, steer=1 * STEER_AMOUNT))

        velocity = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2))

        if len(self.collision_history) != 0:  # Reward for collision
            done = True
            reward = -1
        elif kmh < 30:  # Reward for driving too slow
            done = False
            reward = -0.005
        else:  # Reward for remaining speed
            done = False
            reward = 0.005

        if self.episode_start + EPISODE_SECONDS < time.time():
            done = True

        return self.camera_stream, reward, done

    def process_image(self, img):
        i = np.array(img.raw_data)
        i = i.reshape((IMAGE_DIMENSIONS[1], IMAGE_DIMENSIONS[0], 4))
        i = i[:, :, :3]
        if SHOW_PREVIEW:
            cv2.imshow("", i)
            cv2.waitKey(1)
        self.camera_stream = i

    def process_collision(self, event):
        self.collision_history.append(event)


class Agent:
    global TIMESTAMP

    def __init__(self):
        self.model = self.create_model((IMAGE_DIMENSIONS[1], IMAGE_DIMENSIONS[0], 3))
        self.target_model = self.create_model((IMAGE_DIMENSIONS[1], IMAGE_DIMENSIONS[0], 3))
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        # self.graph = tf.Graph()
        self.tensorboard = ModifiedTensorBoard(log_dir=LOG_PATH + "/" + TIMESTAMP)

        self.last_log = 0
        self.update_counter = 0
        self.terminate = False
        self.initialized = False

    def create_model(self, input_shape):
        base_model = Sequential()
        base_model.add(Conv2D(64, (3, 3), input_shape=input_shape, padding="same"))
        base_model.add(Activation("relu"))
        base_model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding="same"))
        base_model.add(Conv2D(64, (3, 3), input_shape=input_shape, padding="same"))
        base_model.add(Activation("relu"))
        base_model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding="same"))
        base_model.add(Conv2D(64, (3, 3), input_shape=input_shape, padding="same"))
        base_model.add(Activation("relu"))
        base_model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding="same"))
        base_model.add(Flatten())

        '''
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(3, activation="linear")(x)
        '''

        predictions = Dense(3, activation="linear")(base_model.output)
        model = Model(inputs=base_model.inputs, outputs=predictions)
        model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=LEARNING_RATE), metrics=["accuracy"])
        return model

    def train(self):
        if len(self.replay_memory) < REPLAY_MEMORY_MINIMUM:
            return

        mini_batch = random.sample(self.replay_memory, MINI_BATCH_SIZE)

        current_states = np.array([transition[0] for transition in mini_batch]) / 255
        # with self.graph.as_default():
        current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE)
        new_states = np.array([transition[3] for transition in mini_batch]) / 255
        # with self.graph.as_default():
        future_qs_list = self.target_model.predict(new_states, PREDICTION_BATCH_SIZE)

        X = []
        y = []

        for index, (current_state, action, reward, new_state, done) in enumerate(mini_batch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        log = False
        if self.tensorboard.step > self.last_log:
            log = True
            self.last_log = self.tensorboard.step

        # with self.graph.as_default():
        self.model.fit(np.array(X) / 255, np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False,
                       callbacks=[self.tensorboard] if log else None)

        if log:
            self.update_counter += 1

        if self.update_counter > TARGET_UPDATE_INTERVAL:
            self.target_model.set_weights(self.model.get_weights())
            self.update_counter = 0

    def update_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]

    def train_loop(self):
        X = np.random.uniform(size=(1, IMAGE_DIMENSIONS[1], IMAGE_DIMENSIONS[0], 3)).astype(np.float32)
        y = np.random.uniform(size=(1, 3)).astype(np.float32)

        # with self.graph.as_default():
        self.model.fit(X, y, verbose=False, batch_size=1)

        self.initialized = True

        while True:
            if self.terminate:
                return
            else:
                self.train()
                time.sleep(0.01)


class ModifiedTensorBoard(TensorBoard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)

    def set_model(self, model):
        pass

    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, _):
        pass

    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()


if __name__ == "__main__":
    import_carla()
    import carla

    load_config()
    main()
