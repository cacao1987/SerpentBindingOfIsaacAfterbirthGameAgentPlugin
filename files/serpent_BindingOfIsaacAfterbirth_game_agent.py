from serpent.game_agent import GameAgent

import serpent.cv
import serpent.utilities

from serpent.frame_grabber import FrameGrabber

from serpent.input_controller import KeyboardKey

from serpent.machine_learning.reinforcement_learning.ddqn import DDQN
from serpent.machine_learning.reinforcement_learning.keyboard_mouse_action_space import KeyboardMouseActionSpace

from .helpers.frame_processing import frame_to_hearts

import time
import sys
import collections
import os

import gc

import numpy as np

import skimage.io
import skimage.filters
import skimage.morphology
import skimage.measure
import skimage.draw
import skimage.segmentation
import skimage.color

import pyperclip

from datetime import datetime


class SerpentBindingOfIsaacAfterbirthGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play

        self.game_state = None
        self._reset_game_state()

    @property
    def bosses(self):
        return {
            "MONSTRO": "1010"
        }

    def setup_play(self):
        input_mapping = {
            "W": [KeyboardKey.KEY_W],
            "A": [KeyboardKey.KEY_A],
            "S": [KeyboardKey.KEY_S],
            "D": [KeyboardKey.KEY_D],
            "WA": [KeyboardKey.KEY_W, KeyboardKey.KEY_A],
            "WD": [KeyboardKey.KEY_W, KeyboardKey.KEY_D],
            "SA": [KeyboardKey.KEY_S, KeyboardKey.KEY_A],
            "SD": [KeyboardKey.KEY_S, KeyboardKey.KEY_D],
            "UP": [KeyboardKey.KEY_UP],
            "LEFT": [KeyboardKey.KEY_LEFT],
            "DOWN": [KeyboardKey.KEY_DOWN],
            "RIGHT": [KeyboardKey.KEY_RIGHT]
        }

        self.key_mapping = {
            KeyboardKey.KEY_W.name: "MOVE UP",
            KeyboardKey.KEY_A.name: "MOVE LEFT",
            KeyboardKey.KEY_S.name: "MOVE DOWN",
            KeyboardKey.KEY_D.name: "MOVE RIGHT",
            KeyboardKey.KEY_UP.name: "SHOOT UP",
            KeyboardKey.KEY_LEFT.name: "SHOOT LEFT",
            KeyboardKey.KEY_DOWN.name: "SHOOT DOWN",
            KeyboardKey.KEY_RIGHT.name: "SHOOT RIGHT",
        }

        movement_action_space = KeyboardMouseActionSpace(
            directional_keys=[None, "W", "A", "S", "D", "WA", "WD", "SA", "SD"]
        )

        projectile_action_space = KeyboardMouseActionSpace(
            projectile_keys=[None, "UP", "LEFT", "DOWN", "RIGHT"]
        )

        movement_model_file_path = "datasets/binding_of_isaac_movement_dqn_0_1_.h5".replace("/", os.sep)

        self.dqn_movement = DDQN(
            model_file_path=movement_model_file_path if os.path.isfile(movement_model_file_path) else None,
            input_shape=(100, 100, 4),
            input_mapping=input_mapping,
            action_space=movement_action_space,
            replay_memory_size=5000,
            max_steps=1000000,
            observe_steps=1000,
            batch_size=32,
            initial_epsilon=1,
            final_epsilon=0.01,
            override_epsilon=False
        )

        projectile_model_file_path = "datasets/binding_of_isaac_projectile_dqn_0_1_.h5".replace("/", os.sep)

        self.dqn_projectile = DDQN(
            model_file_path=projectile_model_file_path if os.path.isfile(projectile_model_file_path) else None,
            input_shape=(100, 100, 4),
            input_mapping=input_mapping,
            action_space=projectile_action_space,
            replay_memory_size=5000,
            max_steps=1000000,
            observe_steps=1000,
            batch_size=32,
            initial_epsilon=1,
            final_epsilon=0.01,
            override_epsilon=False
        )

        if sys.platform in ["linux", "linux2"]:
            pyperclip.set_clipboard("xsel")

        pyperclip.copy(f"goto s.boss.{self.bosses['MONSTRO']}")

    def handle_play(self, game_frame):
        gc.disable()

        if self.dqn_movement.first_run:
            self._goto_boss()

            self.dqn_movement.first_run = False
            self.dqn_projectile.first_run = False

            return None

        hearts = frame_to_hearts(game_frame.frame, self.game)

        # Check for Curse of Unknown
        if not len(hearts):
            self.input_controller.tap_key(KeyboardKey.KEY_R, duration=1.5)
            self._goto_boss()

            return None

        self.game_state["health"].appendleft(24 - hearts.count(None))
        self.game_state["boss_health"].appendleft(self._get_boss_health(game_frame))

        if self.dqn_movement.frame_stack is None:
            pipeline_game_frame = FrameGrabber.get_frames(
                [0],
                frame_shape=(self.game.frame_height, self.game.frame_width),
                frame_type="PIPELINE",
                dtype="float64"
            ).frames[0]

            self.dqn_movement.build_frame_stack(pipeline_game_frame.frame)
            self.dqn_projectile.frame_stack = self.dqn_movement.frame_stack
        else:
            game_frame_buffer = FrameGrabber.get_frames(
                [0, 4, 8, 12],
                frame_shape=(self.game.frame_height, self.game.frame_width),
                frame_type="PIPELINE",
                dtype="float64"
            )

            if self.dqn_movement.mode == "TRAIN":
                reward_movement, reward_projectile = self._calculate_reward()

                self.game_state["run_reward_movement"] += reward_movement
                self.game_state["run_reward_projectile"] += reward_projectile

                self.dqn_movement.append_to_replay_memory(
                    game_frame_buffer,
                    reward_movement,
                    terminal=self.game_state["health"] == 0
                )

                self.dqn_projectile.append_to_replay_memory(
                    game_frame_buffer,
                    reward_projectile,
                    terminal=self.game_state["health"] == 0
                )

                # Every 2000 steps, save latest weights to disk
                if self.dqn_movement.current_step % 2000 == 0:
                    self.dqn_movement.save_model_weights(
                        file_path_prefix=f"datasets/binding_of_isaac_movement"
                    )

                    self.dqn_projectile.save_model_weights(
                        file_path_prefix=f"datasets/binding_of_isaac_projectile"
                    )

                # Every 20000 steps, save weights checkpoint to disk
                if self.dqn_movement.current_step % 20000 == 0:
                    self.dqn_movement.save_model_weights(
                        file_path_prefix=f"datasets/binding_of_isaac_movement",
                        is_checkpoint=True
                    )

                    self.dqn_projectile.save_model_weights(
                        file_path_prefix=f"datasets/binding_of_isaac_projectile",
                        is_checkpoint=True
                    )
            elif self.dqn_movement.mode == "RUN":
                self.dqn_movement.update_frame_stack(game_frame_buffer)
                self.dqn_projectile.update_frame_stack(game_frame_buffer)

            run_time = datetime.now() - self.started_at

            serpent.utilities.clear_terminal()

            print(f"SESSION RUN TIME: {run_time.days} days, {run_time.seconds // 3600} hours, {(run_time.seconds // 60) % 60} minutes, {run_time.seconds % 60} seconds")
            print("")

            print("MOVEMENT NEURAL NETWORK:\n")
            self.dqn_movement.output_step_data()

            print("")
            print("PROJECTILE NEURAL NETWORK:\n")
            self.dqn_projectile.output_step_data()

            print("")
            print(f"CURRENT RUN: {self.game_state['current_run']}")
            print(f"CURRENT RUN REWARD: {round(self.game_state['run_reward_movement'] + self.game_state['run_reward_projectile'], 2)}")
            print(f"CURRENT RUN PREDICTED ACTIONS: {self.game_state['run_predicted_actions']}")
            print(f"CURRENT HEALTH: {self.game_state['health'][0]}")
            print(f"CURRENT BOSS HEALTH: {self.game_state['boss_health'][0]}")
            print("")
            print(f"LAST RUN DURATION: {self.game_state['last_run_duration']} seconds")

            print("")
            print(f"RECORD TIME ALIVE: {self.game_state['record_time_alive'].get('value')} seconds (Run {self.game_state['record_time_alive'].get('run')}, {'Predicted' if self.game_state['record_time_alive'].get('predicted') else 'Training'}, Boss HP {self.game_state['record_time_alive'].get('boss_hp')})")
            print(f"RECORD BOSS HP: {self.game_state['record_boss_hp'].get('value')} (Run {self.game_state['record_boss_hp'].get('run')}, {'Predicted' if self.game_state['record_boss_hp'].get('predicted') else 'Training'}, Time Alive {self.game_state['record_boss_hp'].get('time_alive')} seconds)")
            print("")

            print(f"RANDOM AVERAGE TIME ALIVE: {self.game_state['random_time_alive']} seconds")
            print(f"RANDOM AVERAGE BOSS HP: {self.game_state['random_boss_hp']}")

            is_boss_dead = self._is_boss_dead(self.game_frame_buffer.previous_game_frame)

            if self.game_state["health"][1] <= 0 or is_boss_dead:
                serpent.utilities.clear_terminal()
                timestamp = datetime.utcnow()

                gc.enable()
                gc.collect()
                gc.disable()

                timestamp_delta = timestamp - self.game_state["run_timestamp"]
                self.game_state["last_run_duration"] = timestamp_delta.seconds

                if self.dqn_movement.mode in ["TRAIN", "RUN"]:
                    # Check for Records
                    if self.game_state["last_run_duration"] > self.game_state["record_time_alive"].get("value", 0):
                        self.game_state["record_time_alive"] = {
                            "value": self.game_state["last_run_duration"],
                            "run": self.game_state["current_run"],
                            "predicted": self.dqn_movement.mode == "RUN",
                            "boss_hp": self.game_state["boss_health"][0]
                        }

                    if self.game_state["boss_health"][0] < self.game_state["record_boss_hp"].get("value", 1000):
                        self.game_state["record_boss_hp"] = {
                            "value": self.game_state["boss_health"][0],
                            "run": self.game_state["current_run"],
                            "predicted": self.dqn_movement.mode == "RUN",
                            "time_alive": self.game_state["last_run_duration"]
                        }
                else:
                    self.game_state["random_time_alives"].append(self.game_state["last_run_duration"])
                    self.game_state["random_boss_hps"].append(self.game_state["boss_health"][0])

                    self.game_state["random_time_alive"] = np.mean(self.game_state["random_time_alives"])
                    self.game_state["random_boss_hp"] = np.mean(self.game_state["random_boss_hps"])

                self.game_state["current_run_steps"] = 0

                self.input_controller.handle_keys([])
                self.input_controller.tap_key(KeyboardKey.KEY_R, duration=1.5)

                if self.dqn_movement.mode == "TRAIN":
                    for i in range(16):
                        serpent.utilities.clear_terminal()
                        print(f"TRAINING ON MINI-BATCHES: {i + 1}/16")
                        print(f"NEXT RUN: {self.game_state['current_run'] + 1} {'- AI RUN' if (self.game_state['current_run'] + 1) % 20 == 0 else ''}")

                        self.dqn_movement.train_on_mini_batch()
                        self.dqn_projectile.train_on_mini_batch()

                self.game_state["boss_skull_image"] = None

                self.game_state["run_timestamp"] = datetime.utcnow()
                self.game_state["current_run"] += 1
                self.game_state["run_reward_movement"] = 0
                self.game_state["run_reward_projectile"] = 0
                self.game_state["run_predicted_actions"] = 0
                self.game_state["health"] = collections.deque(np.full((8,), 6), maxlen=8)
                self.game_state["boss_health"] = collections.deque(np.full((8,), 654), maxlen=8)

                if self.dqn_movement.mode in ["TRAIN", "RUN"]:
                    if self.game_state["current_run"] > 0 and self.game_state["current_run"] % 100 == 0:
                        self.dqn_movement.update_target_model()
                        self.dqn_projectile.update_target_model()

                    if self.game_state["current_run"] > 0 and self.game_state["current_run"] % 20 == 0:
                        self.dqn_movement.enter_run_mode()
                        self.dqn_projectile.enter_run_mode()
                    else:
                        self.dqn_movement.enter_train_mode()
                        self.dqn_projectile.enter_train_mode()

                self._goto_boss()

                return None

        self.dqn_movement.pick_action()
        self.dqn_movement.generate_action()

        self.dqn_projectile.pick_action(action_type=self.dqn_movement.current_action_type)
        self.dqn_projectile.generate_action()

        movement_keys = self.dqn_movement.get_input_values()
        projectile_keys = self.dqn_projectile.get_input_values()

        print("")
        print(" + ".join(list(map(lambda k: self.key_mapping.get(k.name), movement_keys + projectile_keys))))

        self.input_controller.handle_keys(movement_keys + projectile_keys)

        if self.dqn_movement.current_action_type == "PREDICTED":
            self.game_state["run_predicted_actions"] += 1

        self.dqn_movement.erode_epsilon(factor=2)
        self.dqn_projectile.erode_epsilon(factor=2)

        self.dqn_movement.next_step()
        self.dqn_projectile.next_step()

        self.game_state["current_run_steps"] += 1

    def _reset_game_state(self):
        self.game_state = {
            "health": collections.deque(np.full((8,), 6), maxlen=8),
            "boss_health": collections.deque(np.full((8,), 654), maxlen=8),
            "boss_skull_image": None,
            "current_run": 1,
            "current_run_steps": 0,
            "run_reward_movement": 0,
            "run_reward_projectile": 0,
            "run_future_rewards": 0,
            "run_predicted_actions": 0,
            "run_timestamp": datetime.utcnow(),
            "last_run_duration": 0,
            "record_time_alive": dict(),
            "record_boss_hp": dict(),
            "random_time_alive": None,
            "random_time_alives": list(),
            "random_boss_hp": None,
            "random_boss_hps": list()
        }

    def _goto_boss(self):
        self.input_controller.tap_key(KeyboardKey.KEY_SPACE)
        time.sleep(1)
        self.input_controller.tap_key(KeyboardKey.KEY_GRAVE)
        time.sleep(0.5)

        self.input_controller.tap_keys([KeyboardKey.KEY_LEFT_CTRL, KeyboardKey.KEY_V])

        self.input_controller.tap_key(KeyboardKey.KEY_ENTER)
        self.input_controller.tap_key(KeyboardKey.KEY_ENTER)
        time.sleep(0.5)
        self.input_controller.tap_key(KeyboardKey.KEY_ENTER)
        time.sleep(0.5)
        self.input_controller.tap_key(KeyboardKey.KEY_ENTER)
        time.sleep(0.2)

    def _get_boss_health(self, game_frame):
        gray_boss_health_bar = serpent.cv.extract_region_from_image(
            game_frame.grayscale_frame,
            self.game.screen_regions["HUD_BOSS_HP"]
        )

        try:
            threshold = skimage.filters.threshold_otsu(gray_boss_health_bar)
        except ValueError:
            threshold = 1

        bw_boss_health_bar = gray_boss_health_bar > threshold

        return bw_boss_health_bar[bw_boss_health_bar > 0].size

    def _is_boss_dead(self, game_frame):
        gray_boss_skull = serpent.cv.extract_region_from_image(
            game_frame.grayscale_frame,
            self.game.screen_regions["HUD_BOSS_SKULL"]
        )

        if self.game_state["boss_skull_image"] is None:
            self.game_state["boss_skull_image"] = gray_boss_skull

        is_dead = False

        if skimage.measure.compare_ssim(gray_boss_skull, self.game_state["boss_skull_image"]) < 0.5:
            is_dead = True

        self.game_state["boss_skull_image"] = gray_boss_skull

        return is_dead

    def _calculate_reward(self):
        reward_movement = 0
        reward_projectile = 0

        reward_movement += (-1 if self.game_state["health"][0] < self.game_state["health"][1] else 0.05)
        reward_projectile += (1 if self.game_state["boss_health"][0] < self.game_state["boss_health"][3] else -0.05)

        return reward_movement, reward_projectile
