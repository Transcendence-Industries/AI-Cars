import os
import sys
import glob
import time
import math
from collections import deque

import cv2
import numpy as np
from keras.models import load_model

CARLA_PATH = "/home/mainframe/Workspace/Tools/Carla/PythonAPI/carla"
MODEL_PATH = "/home/mainframe/Workspace/Logs/Node-10/*/model"
MAINFRAME_IP = "127.0.0.1"

EPISODE_SECONDS = 10

SHOW_PREVIEW = True
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


def main():
    # Prepare model and environment
    model = load_model(MODEL_PATH)
    env = Env()
    fps_counter = deque(maxlen=FPS)
    model.predict(np.ones((1, IMAGE_DIMENSIONS[1], IMAGE_DIMENSIONS[0], 3)))

    while True:
        current_state = env.reset()
        env.collision_history = []
        done = False

        while True:
            step_start = time.time()

            if SHOW_PREVIEW:
                cv2.imshow("", current_state)
                cv2.waitKey(1)

            # Predict and execute action
            qs = model.predict(np.array(current_state).reshape(-1, *current_state.shape) / 255)[0]
            action = np.argmax(qs)
            new_state, reward, done = env.step(action)
            current_state = new_state

            if done:
                break

            frame_time = time.time() - step_start
            fps_counter.append(frame_time)

        for actor in env.actor_list:
            actor.destroy()


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

        vehicle_sp = self.world.get_map().get_spawn_points()[0]
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
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, brake=0, steer=0))
        elif action == 1:  # Steer left
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, brake=0, steer=-1 * STEER_AMOUNT))
        elif action == 2:  # Steer right
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, brake=0, steer=1 * STEER_AMOUNT))

        velocity = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2))

        if len(self.collision_history) != 0:  # Reward for collision
            done = True
            reward = -200
        elif kmh < 50:  # Reward for driving too slow
            done = False
            reward = -1
        else:  # Reward for remaining speed
            done = False
            reward = 1

        if self.episode_start + EPISODE_SECONDS < time.time():
            done = True

        return self.camera_stream, reward, done

    def process_image(self, img):
        i = np.array(img.raw_data)
        i = i.reshape((IMAGE_DIMENSIONS[1], IMAGE_DIMENSIONS[0], 4))
        i = i[:, :, :3]
        self.camera_stream = i

    def process_collision(self, event):
        self.collision_history.append(event)


if __name__ == "__main__":
    import_carla()
    import carla

    main()
